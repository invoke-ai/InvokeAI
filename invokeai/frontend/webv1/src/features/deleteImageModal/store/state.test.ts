/**
 * Regression tests for the post-delete selection logic in the image delete flow.
 *
 * The bug (PR #9163 review): ``handleDeletions`` cleared the viewer selection
 * (``imageSelected(null)``) whenever the deletion intersected the multi-selection but the
 * *displayed* item was not among the deleted names — e.g. a video displayed while only
 * images were deleted, or a hover-delete of a non-displayed selected image. It also treated
 * every *requested* name as deleted, ignoring the server's ``deleted_images`` response, so a
 * partial failure could jump the selection away from (or past) an image that still exists.
 *
 * The fix mirrors deleteVideoModal/store/state.ts: only server-confirmed deletions count,
 * and a surviving displayed item stays selected.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('services/api/endpoints/images', () => ({
  imagesApi: {
    endpoints: {
      deleteImages: {
        initiate: vi.fn((arg: { image_names: string[] }) => ({
          type: 'imagesApi/deleteImages',
          image_names: arg.image_names,
        })),
      },
    },
  },
}));

vi.mock('features/gallery/store/gallerySelectors', () => ({
  selectLastSelectedItem: vi.fn(),
}));

vi.mock('features/gallery/store/gallerySlice', () => ({
  imageSelected: vi.fn((payload: string | null) => ({ type: 'gallery/imageSelected', payload })),
  selectionChanged: vi.fn((payload: string[]) => ({ type: 'gallery/selectionChanged', payload })),
}));

vi.mock('features/system/store/systemSlice', () => ({
  selectSystemShouldConfirmOnDelete: vi.fn(() => false),
}));

// The canvas/ref-image usage sweeps aren't under test — give them empty state.
vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasSlice: vi.fn(() => ({ controlLayers: { entities: [] }, rasterLayers: { entities: [] } })),
}));

vi.mock('features/controlLayers/store/refImagesSlice', () => ({
  refImageImageChanged: vi.fn(),
  selectReferenceImageEntities: vi.fn(() => []),
  selectRefImagesSlice: vi.fn(() => ({ entities: [] })),
}));

// Keep the real pickSelectionAfterDelete (its neighbour-picking is part of the behavior under
// test) but stub the cache selector, which would otherwise need a live RTK Query store.
vi.mock('features/gallery/store/selectCachedGalleryItemNames', async (importOriginal) => {
  const actual = await importOriginal<object>();
  return { ...actual, selectCachedGalleryItemNames: vi.fn() };
});

import type { AppStore } from 'app/store/store';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { selectCachedGalleryItemNames } from 'features/gallery/store/selectCachedGalleryItemNames';

import { handleDeletions } from './state';

/**
 * `selectionDuringDelete` re-points the store's selection when the delete request is issued,
 * standing in for the user selecting something else while it is in flight. The rest of
 * handleDeletions then runs against that newer selection, as it does in the app.
 *
 * It writes `currentSelection` directly rather than dispatching, so the simulated gesture does not
 * land in `dispatched` and cannot be mistaken for the production write under test. The seam is
 * only equivalent to a real mid-flight click because neither modal reads state between issuing the
 * request and the post-await block; anything added in between would need a real dispatch here.
 */
const buildStore = (selection: string[], failingNames: Set<string>, selectionDuringDelete?: string[]) => {
  const dispatched: unknown[] = [];
  let currentSelection = selection;
  const dispatch = vi.fn((action: unknown) => {
    dispatched.push(action);
    const typed = action as { type?: string; image_names?: string[] };
    if (typed?.type === 'imagesApi/deleteImages') {
      return {
        unwrap: () => {
          if (selectionDuringDelete) {
            currentSelection = selectionDuringDelete;
          }
          return Promise.resolve({
            deleted_images: (typed.image_names ?? []).filter((name) => !failingNames.has(name)),
            affected_boards: [],
          });
        },
      };
    }
    return action;
  });
  const getState = vi.fn(() => ({ gallery: { selection: currentSelection }, nodes: { present: { nodes: [] } } }));
  return { store: { dispatch, getState } as unknown as AppStore, dispatched };
};

/**
 * Every write to the selection, in order and raw — so an expectation pins the whole payload *and*
 * that there was exactly one write. Returning just the first match let a stray second dispatch
 * (which is what the user would actually end up looking at) pass unnoticed.
 *
 * The two branches use different actions deliberately: advancing to a neighbour *picks* an item
 * (`imageSelected`), while keeping the displayed item and dropping the deleted ones from the
 * multi-selection is a *mutation* (`selectionChanged`), which the viewer does not treat as the
 * user asking to see anything.
 */
const getSelectionWrites = (dispatched: unknown[]) =>
  dispatched.filter(
    (candidate): candidate is { type: string; payload: string | string[] | null } =>
      !!candidate &&
      typeof candidate === 'object' &&
      ((candidate as { type?: string }).type === 'gallery/imageSelected' ||
        (candidate as { type?: string }).type === 'gallery/selectionChanged')
  );

describe('handleDeletions selection behavior', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(selectCachedGalleryItemNames).mockReturnValue(['a.png', 'b.png', 'c.mp4']);
  });

  it('keeps viewing a displayed video when only images from a mixed selection are deleted', async () => {
    // Video c.mp4 is displayed; the user deletes image a.png from the mixed selection.
    vi.mocked(selectLastSelectedItem).mockReturnValue('c.mp4');
    const { store, dispatched } = buildStore(['a.png', 'c.mp4'], new Set());

    await handleDeletions(['a.png'], store);

    // Before the fix this dispatched imageSelected(null) and dropped the viewer to its
    // empty state even though the displayed video still exists.
    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/selectionChanged', payload: ['c.mp4'] }]);
  });

  it('keeps viewing the displayed image on hover-delete of another selected image, without re-picking it', async () => {
    // Multi-selection [a, b] with b displayed; the hover delete button deletes only a.
    vi.mocked(selectLastSelectedItem).mockReturnValue('b.png');
    const { store, dispatched } = buildStore(['a.png', 'b.png'], new Set());

    await handleDeletions(['a.png'], store);

    // The action matters, not just the resulting state (PR #9520 review). `imageSelected('b.png')`
    // would leave the very same selection, but it is the action that means "the user picked this",
    // and the viewer answers a pick by lifting an in-progress generation's overlay off the item for
    // two seconds. Deleting `a` must not flash `b` over a running render.
    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/selectionChanged', payload: ['b.png'] }]);
    expect(imageSelected).not.toHaveBeenCalled();
  });

  it('does not move the selection when the displayed image fails to delete', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.png');
    const { store, dispatched } = buildStore(['a.png'], new Set(['a.png']));

    await handleDeletions(['a.png'], store);

    expect(getSelectionWrites(dispatched), 'a failed delete must not advance the selection').toEqual([]);
  });

  it('keeps a surviving (failed-delete) neighbour as the replacement candidate', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.png');
    // Batch delete of a + b: a (the displayed item) deletes fine, b fails and still exists.
    const { store, dispatched } = buildStore(['a.png'], new Set(['b.png']));

    await handleDeletions(['a.png', 'b.png'], store);

    // If b.png were treated as deleted, the selection would skip to c.mp4.
    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/imageSelected', payload: 'b.png' }]);
  });

  it('advances to the nearest surviving neighbour when everything requested is deleted', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('b.png');
    const { store, dispatched } = buildStore(['b.png'], new Set());

    await handleDeletions(['b.png'], store);

    // The displayed item is gone, so the viewer really does move to a different image: that is a
    // pick, and revealing it over a running generation is the point.
    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/imageSelected', payload: 'a.png' }]);
  });

  it('keeps surviving siblings of a larger multi-selection', async () => {
    // No race here: [a, b, c] with c displayed, delete a. Collapsing onto the displayed item used
    // to deselect b as collateral, which the branch's own comment never claimed to do.
    vi.mocked(selectLastSelectedItem).mockReturnValue('c.mp4');
    const { store, dispatched } = buildStore(['a.png', 'b.png', 'c.mp4'], new Set());

    await handleDeletions(['a.png'], store);

    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/selectionChanged', payload: ['b.png', 'c.mp4'] }]);
  });

  it('leaves a selection made while the delete was in flight alone', async () => {
    // The branch decides on a snapshot taken before the request, so by the time it runs the user
    // may have selected something else. Collapsing onto the snapshot would discard that pick *and*
    // move the active item back — which the viewer publishes as a change of active item and
    // reveals, the very flash the mutation action avoids.
    vi.mocked(selectLastSelectedItem).mockReturnValue('b.png');
    const { store, dispatched } = buildStore(['a.png', 'b.png'], new Set(), ['a.png', 'b.png', 'c.mp4']);

    await handleDeletions(['a.png'], store);

    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/selectionChanged', payload: ['b.png', 'c.mp4'] }]);
    expect(imageSelected).not.toHaveBeenCalled();
  });

  it('falls back to the surviving displayed item when the newer selection is all deleted', async () => {
    // Same race, but everything the user selected meanwhile went away: there is nothing to keep,
    // so the viewer stays on the item that survived rather than emptying out.
    vi.mocked(selectLastSelectedItem).mockReturnValue('b.png');
    const { store, dispatched } = buildStore(['a.png', 'b.png'], new Set(), ['a.png']);

    await handleDeletions(['a.png'], store);

    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/selectionChanged', payload: ['b.png'] }]);
  });
});
