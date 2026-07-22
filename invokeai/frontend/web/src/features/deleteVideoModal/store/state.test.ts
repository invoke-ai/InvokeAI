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
}));

vi.mock('features/nodes/store/nodesSlice', () => ({
  fieldVideoValueChanged: vi.fn((payload: unknown) => ({ type: 'nodes/fieldVideoValueChanged', payload })),
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
import { videosApi } from 'services/api/endpoints/videos';

import { handleDeletions } from './state';

const buildVideoFieldNode = (nodeId: string, videoName: string) => ({
  type: 'invocation',
  data: {
    id: nodeId,
    inputs: {
      video: { name: 'video', label: '', description: '', value: { video_name: videoName } },
    },
  },
});

const buildStore = (selection: string[], failingNames: Set<string>, nodes: unknown[] = [], rejectAll = false) => {
  const dispatched: unknown[] = [];
  const dispatch = vi.fn((action: unknown) => {
    dispatched.push(action);
    const typed = action as { type?: string; video_names?: string[] };
    if (typed?.type === 'videosApi/deleteVideos') {
      return {
        unwrap: () =>
          rejectAll
            ? Promise.reject(new Error('delete failed'))
            : Promise.resolve({
                deleted_videos: (typed.video_names ?? []).filter((name) => !failingNames.has(name)),
                affected_boards: ['none'],
              }),
      };
    }
    return action;
  });
  const getState = vi.fn(() => ({ gallery: { selection }, nodes: { present: { nodes } } }));
  return { store: { dispatch, getState } as unknown as AppStore, dispatched };
};

const getSelectionChange = (dispatched: unknown[]) =>
  dispatched.find(
    (action): action is { type: string; payload: string | null } =>
      !!action && typeof action === 'object' && (action as { type?: string }).type === 'gallery/imageSelected'
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

    expect(getSelectionChange(dispatched), 'a failed delete must not advance the selection').toBeUndefined();
  });

  it('does not move the selection when the whole batch request fails', async () => {
    vi.mocked(selectLastSelectedItem).mockReturnValue('a.mp4');
    const { store, dispatched } = buildStore(['a.mp4'], new Set(), [], true);

    await handleDeletions(['a.mp4'], store);

    expect(getSelectionChange(dispatched)).toBeUndefined();
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
