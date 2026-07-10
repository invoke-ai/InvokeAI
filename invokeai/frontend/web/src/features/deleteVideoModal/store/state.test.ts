/**
 * Regression tests for the post-delete selection logic in the video delete flow.
 *
 * The bug (PR #9163 review): ``handleDeletions`` advanced the gallery selection as if every
 * requested video was deleted, even when individual ``deleteVideo`` calls failed (403/500).
 * The failed names were included in the "deleted" set passed to ``pickSelectionAfterDelete``,
 * so the Viewer could jump away from a video that still exists — and a surviving neighbour
 * could be skipped as a replacement candidate.
 *
 * The fix tracks which deletions actually resolved and only treats those as deleted.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('services/api/endpoints/videos', () => ({
  videosApi: {
    endpoints: {
      deleteVideo: {
        initiate: vi.fn((arg: { video_name: string }) => ({
          type: 'videosApi/deleteVideo',
          video_name: arg.video_name,
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
}));

vi.mock('features/system/store/systemSlice', () => ({
  selectSystemShouldConfirmOnDelete: vi.fn(() => false),
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

const buildStore = (selection: string[], failingNames: Set<string>) => {
  const dispatched: unknown[] = [];
  const dispatch = vi.fn((action: unknown) => {
    dispatched.push(action);
    const typed = action as { type?: string; video_name?: string };
    if (typed?.type === 'videosApi/deleteVideo') {
      return {
        unwrap: () =>
          typed.video_name && failingNames.has(typed.video_name)
            ? Promise.reject(new Error('delete failed'))
            : Promise.resolve(undefined),
      };
    }
    return action;
  });
  const getState = vi.fn(() => ({ gallery: { selection } }));
  return { store: { dispatch, getState } as unknown as AppStore, dispatched };
};

const getSelectionChange = (dispatched: unknown[]) =>
  dispatched.find(
    (action): action is { type: string; payload: string | null } =>
      !!action && typeof action === 'object' && (action as { type?: string }).type === 'gallery/imageSelected'
  );

describe('handleDeletions selection behavior on partial failure', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(selectCachedGalleryItemNames).mockReturnValue(['a.mp4', 'b.mp4', 'c.png']);
  });

  it('does not move the selection when the displayed video fails to delete', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    const { store, dispatched } = buildStore(['a.mp4'], new Set(['a.mp4']));

    await handleDeletions(['a.mp4'], store);

    expect(getSelectionChange(dispatched), 'a failed delete must not advance the selection').toBeUndefined();
  });

  it('keeps a surviving (failed-delete) neighbour as the replacement candidate', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    // Batch delete of a + b: a (the displayed item) deletes fine, b fails and still exists.
    const { store, dispatched } = buildStore(['a.mp4'], new Set(['b.mp4']));

    await handleDeletions(['a.mp4', 'b.mp4'], store);

    // Before the fix, b.mp4 was excluded as "deleted" and the selection skipped to c.png.
    expect(getSelectionChange(dispatched)?.payload).toBe('b.mp4');
  });

  it('keeps viewing the displayed video when its delete fails but another selected video was deleted', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    const { store, dispatched } = buildStore(['a.mp4', 'b.mp4'], new Set(['a.mp4']));

    await handleDeletions(['a.mp4', 'b.mp4'], store);

    // The multi-selection contained a deleted item (b), so the selection is pruned — but it
    // must land on the still-existing displayed video, not jump to a neighbour.
    expect(getSelectionChange(dispatched)?.payload).toBe('a.mp4');
  });

  it('advances to the nearest surviving neighbour when everything requested is deleted', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('b.mp4');
    const { store, dispatched } = buildStore(['b.mp4'], new Set());

    await handleDeletions(['b.mp4'], store);

    expect(getSelectionChange(dispatched)?.payload).toBe('a.mp4');
    expect(imageSelected).toHaveBeenCalledWith('a.mp4');
  });
});
