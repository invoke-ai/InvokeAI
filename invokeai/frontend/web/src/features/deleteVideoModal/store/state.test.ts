/**
 * Regression tests for the video delete flow.
 *
 * Selection (PR #9163 review): ``handleDeletions`` must only treat server-confirmed
 * deletions as deleted — the Viewer must not jump away from a video whose delete failed
 * (403/500), and a surviving neighbour remains a valid replacement candidate.
 *
 * Batching (PR #9163 review): deletion goes through the batch ``deleteVideos`` endpoint —
 * one request per invocation, not one per video — and its ``deleted_videos`` result is the
 * source of truth for partial failures.
 *
 * Node references (PR #9163 review): workflow nodes take VideoField inputs; references to
 * confirmed-deleted videos are cleared, references to surviving videos are preserved.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('services/api/endpoints/videos', () => ({
  videosApi: {
    endpoints: {
      deleteVideos: {
        initiate: vi.fn((arg: { video_names: string[] }) => ({
          type: 'videosApi/deleteVideos',
          video_names: arg.video_names,
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

vi.mock('features/nodes/store/nodesSlice', () => ({
  fieldVideoValueChanged: vi.fn((payload: unknown) => ({ type: 'nodes/fieldVideoValueChanged', payload })),
}));

vi.mock('features/system/store/systemSlice', () => ({
  selectSystemShouldConfirmOnDelete: vi.fn(() => false),
}));

vi.mock('features/toast/toast', () => ({ toast: vi.fn() }));

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
import { selectSystemShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { toast } from 'features/toast/toast';
import { videosApi } from 'services/api/endpoints/videos';

import { cancelDeletion, confirmDeletion, deleteVideosWithDialog, handleDeletions } from './state';

const buildVideoFieldNode = (nodeId: string, videoName: string) => ({
  type: 'invocation',
  data: {
    id: nodeId,
    inputs: {
      video: { name: 'video', label: '', description: '', value: { video_name: videoName } },
    },
  },
});

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
const buildStore = (
  selection: string[],
  failingNames: Set<string>,
  nodes: unknown[] = [],
  rejectAll = false,
  selectionDuringDelete?: string[]
) => {
  const dispatched: unknown[] = [];
  let currentSelection = selection;
  const dispatch = vi.fn((action: unknown) => {
    dispatched.push(action);
    const typed = action as { type?: string; video_names?: string[] };
    if (typed?.type === 'videosApi/deleteVideos') {
      return {
        unwrap: () => {
          if (selectionDuringDelete) {
            currentSelection = selectionDuringDelete;
          }
          return rejectAll
            ? Promise.reject(new Error('delete failed'))
            : Promise.resolve({
                deleted_videos: (typed.video_names ?? []).filter((name) => !failingNames.has(name)),
                failed_videos: (typed.video_names ?? []).filter((name) => failingNames.has(name)),
                affected_boards: ['none'],
              });
        },
      };
    }
    return action;
  });
  const getState = vi.fn(() => ({ gallery: { selection: currentSelection }, nodes: { present: { nodes } } }));
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

const getVideoFieldChanges = (dispatched: unknown[]) =>
  dispatched.filter(
    (action): action is { type: string; payload: { nodeId: string; fieldName: string; value: unknown } } =>
      !!action && typeof action === 'object' && (action as { type?: string }).type === 'nodes/fieldVideoValueChanged'
  );

describe('handleDeletions batching', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(selectCachedGalleryItemNames).mockReturnValue(['a.mp4', 'b.mp4', 'c.png']);
    vi.mocked(selectLastSelectedItem).mockReturnValue(undefined);
  });

  it('issues a single batch request for a multi-video deletion', async () => {
    const { store } = buildStore([], new Set());

    await handleDeletions(['a.mp4', 'b.mp4'], store);

    expect(videosApi.endpoints.deleteVideos.initiate).toHaveBeenCalledTimes(1);
    expect(videosApi.endpoints.deleteVideos.initiate).toHaveBeenCalledWith(
      { video_names: ['a.mp4', 'b.mp4'] },
      { track: false }
    );
    expect(toast).not.toHaveBeenCalled();
  });
});

describe('handleDeletions selection behavior on partial failure', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(selectCachedGalleryItemNames).mockReturnValue(['a.mp4', 'b.mp4', 'c.png']);
  });

  it('does not move the selection when the displayed video fails to delete', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    const { store, dispatched } = buildStore(['a.mp4'], new Set(['a.mp4']));

    await handleDeletions(['a.mp4'], store);

    expect(getSelectionWrites(dispatched), 'a failed delete must not advance the selection').toEqual([]);
  });

  it('does not move the selection when the whole batch request fails', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    const { store, dispatched } = buildStore(['a.mp4'], new Set(), [], true);

    await handleDeletions(['a.mp4'], store);

    expect(getSelectionWrites(dispatched)).toEqual([]);
  });

  it('keeps a surviving (failed-delete) neighbour as the replacement candidate', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    // Batch delete of a + b: a (the displayed item) deletes fine, b fails and still exists.
    const { store, dispatched } = buildStore(['a.mp4'], new Set(['b.mp4']));

    await handleDeletions(['a.mp4', 'b.mp4'], store);

    // Before the fix, b.mp4 was excluded as "deleted" and the selection skipped to c.png.
    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/imageSelected', payload: 'b.mp4' }]);
    expect(toast).toHaveBeenCalledWith(expect.objectContaining({ status: 'warning' }));
  });

  it('keeps viewing the displayed video when another selected video was deleted, without re-picking it', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    const { store, dispatched } = buildStore(['a.mp4', 'b.mp4'], new Set(['a.mp4']));

    await handleDeletions(['a.mp4', 'b.mp4'], store);

    // The multi-selection contained a deleted item (b), so the selection is pruned — but it
    // must land on the still-existing displayed video, not jump to a neighbour.
    //
    // And it must prune rather than re-pick (PR #9520 review): `imageSelected('a.mp4')` leaves the
    // very same selection, but it is the action that means "the user picked this", which makes the
    // viewer lift an in-progress generation's overlay off the video for two seconds. Deleting some
    // other video is not a request to look at this one.
    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/selectionChanged', payload: ['a.mp4'] }]);
    expect(imageSelected).not.toHaveBeenCalled();
  });

  it('advances to the nearest surviving neighbour when everything requested is deleted', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('b.mp4');
    const { store, dispatched } = buildStore(['b.mp4'], new Set());

    await handleDeletions(['b.mp4'], store);

    // The displayed item is gone, so the viewer really does move to a different video: that is a
    // pick, and revealing it over a running generation is the point.
    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/imageSelected', payload: 'a.mp4' }]);
    expect(imageSelected).toHaveBeenCalledWith('a.mp4');
  });

  it('leaves a selection made while the delete was in flight alone', async () => {
    // The branch decides on a snapshot taken before the request, so by the time it runs the user
    // may have selected something else. Collapsing onto the snapshot would discard that pick *and*
    // move the active item back — which the viewer publishes as a change of active item and
    // reveals, the very flash the mutation action avoids.
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    const { store, dispatched } = buildStore(['a.mp4', 'b.mp4'], new Set(['a.mp4']), [], false, [
      'a.mp4',
      'b.mp4',
      'c.png',
    ]);

    await handleDeletions(['a.mp4', 'b.mp4'], store);

    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/selectionChanged', payload: ['a.mp4', 'c.png'] }]);
    expect(imageSelected).not.toHaveBeenCalled();
  });

  it('falls back to the surviving displayed video when the newer selection is all deleted', async () => {
    // Same race, but everything the user selected meanwhile went away. Without the fallback this
    // dispatches selectionChanged([]) and drops the viewer to its empty-state placeholder while a
    // surviving video is still on screen — the original #9163 bug this file exists to guard.
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    const { store, dispatched } = buildStore(['a.mp4'], new Set(['a.mp4']), [], false, ['b.mp4']);

    await handleDeletions(['a.mp4', 'b.mp4'], store);

    expect(getSelectionWrites(dispatched)).toEqual([{ type: 'gallery/selectionChanged', payload: ['a.mp4'] }]);
  });
});

describe('handleDeletions node VideoField cleanup', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(selectCachedGalleryItemNames).mockReturnValue(['a.mp4', 'b.mp4']);
    vi.mocked(selectLastSelectedItem).mockReturnValue(undefined);
  });

  it('clears VideoField inputs that reference a confirmed-deleted video', async () => {
    const nodes = [buildVideoFieldNode('n1', 'a.mp4'), buildVideoFieldNode('n2', 'b.mp4')];
    const { store, dispatched } = buildStore([], new Set(), nodes);

    await handleDeletions(['a.mp4'], store);

    const changes = getVideoFieldChanges(dispatched);
    expect(changes).toHaveLength(1);
    expect(changes[0]?.payload).toEqual({ nodeId: 'n1', fieldName: 'video', value: undefined });
  });

  it('preserves VideoField inputs for videos whose deletion failed', async () => {
    const nodes = [buildVideoFieldNode('n1', 'a.mp4'), buildVideoFieldNode('n2', 'b.mp4')];
    const { store, dispatched } = buildStore([], new Set(['b.mp4']), nodes);

    await handleDeletions(['a.mp4', 'b.mp4'], store);

    const changes = getVideoFieldChanges(dispatched);
    expect(changes).toHaveLength(1);
    expect(changes[0]?.payload.nodeId).toBe('n1');
  });

  it('preserves all VideoField inputs when the whole batch request fails', async () => {
    const nodes = [buildVideoFieldNode('n1', 'a.mp4')];
    const { store, dispatched } = buildStore([], new Set(), nodes, true);

    await handleDeletions(['a.mp4'], store);

    expect(getVideoFieldChanges(dispatched)).toHaveLength(0);
  });
});

describe('delete dialog dismissal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(selectCachedGalleryItemNames).mockReturnValue(['a.mp4']);
    vi.mocked(selectLastSelectedItem).mockReturnValue(undefined);
  });

  it('routes generic dialog closure through cancellation so callers settle', () => {
    const source = readFileSync(fileURLToPath(new URL('../components/DeleteVideoModal.tsx', import.meta.url)), 'utf8');

    expect(source).toContain('onClose={api.cancel}');
    expect(source).not.toContain('onClose={api.close}');
  });

  it('rejects the caller when the dialog is dismissed without confirming', async () => {
    vi.mocked(selectSystemShouldConfirmOnDelete).mockReturnValue(true);
    const { store } = buildStore([], new Set());

    const promise = deleteVideosWithDialog(['a.mp4'], store);
    cancelDeletion();

    await expect(promise).rejects.toBe('User canceled');
    expect(videosApi.endpoints.deleteVideos.initiate).not.toHaveBeenCalled();
  });

  it('resolves a confirmed deletion even though the accept path also fires onClose (cancel)', async () => {
    // ConfirmationAlertDialog's accept button invokes acceptCallback() and then
    // onClose() synchronously, without awaiting the callback. With onClose bound to
    // cancel, that cancel must be a no-op after confirm — not a rejection of the
    // deletion the user just accepted.
    vi.mocked(selectSystemShouldConfirmOnDelete).mockReturnValue(true);
    const { store } = buildStore([], new Set());

    const promise = deleteVideosWithDialog(['a.mp4'], store);
    const confirming = confirmDeletion(store);
    cancelDeletion();
    await confirming;

    await expect(promise).resolves.toBeUndefined();
    expect(videosApi.endpoints.deleteVideos.initiate).toHaveBeenCalledTimes(1);
  });
});
