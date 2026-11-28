import type { GalleryItemRef, GalleryVideoItem } from '@features/gallery';
import type { VideoUiAdapter } from '@features/video';
import type { OpenResult } from '@workbench/workspace/openIntent';
import type { ReactNode } from 'react';

import { claimGalleryNavigationSequence } from '@features/gallery/contracts';
import { getGalleryRevealRequest } from '@features/gallery/core/selection';
import {
  clearVideoSpanPlaybackState,
  consumeVideoSpanPlaybackRequest,
  getVideoSpanPlaybackRequest,
  publishVideoSpanPlaybackState,
} from '@workbench/widgets/preview/spanPlaybackRequest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoUiAdapterProvider } from './VideoUiAdapter';

/**
 * Verify ordering across Preview placement, selection, and span requests; failed play must not change persisted
 * selection.
 */

let adapter: VideoUiAdapter;
let openResult: OpenResult;
let activeProjectId: string;

const open = vi.fn(() => openResult);
const selectItem = vi.fn();
const reportError = vi.fn();
const revealGalleryItem = vi.fn(
  (_context: unknown, _ref: GalleryItemRef, _ticket: { projectId: string; sequence: number }) => Promise.resolve()
);
/** The store's projects, read by `queries.getProject` at call time. */
let storedProjects: Record<string, unknown> = {};

vi.mock('@features/video', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  VideoUiProvider: ({ adapter: next, children }: { adapter: VideoUiAdapter; children: ReactNode }) => {
    adapter = next;
    return children;
  },
}));
const galleryQueries = vi.hoisted(() => ({
  canAddToCachedGalleryBoard: vi.fn(() => true),
  ensureGalleryBoardDestinations: vi.fn((): Promise<void> | null => null),
}));
vi.mock('@features/gallery/queries', () => ({
  canAddToCachedGalleryBoard: galleryQueries.canAddToCachedGalleryBoard,
  ensureGalleryBoardDestinations: galleryQueries.ensureGalleryBoardDestinations,
  invalidateGallery: vi.fn(),
  invalidateGalleryBoardDestinations: vi.fn(() => Promise.resolve()),
}));
vi.mock('@workbench/image-actions/revealGalleryItem', () => ({ revealGalleryItem }));
vi.mock('@tanstack/react-query', () => ({ useQueryClient: () => ({}) }));
vi.mock('@workbench/settings/store', () => ({ useWorkbenchPreferenceSelector: () => false }));
vi.mock('@workbench/workspace/useWorkspaceOpen', () => ({ useWorkspaceOpen: () => open }));
/** The active project's widget instances, read by the adapter's project selectors. */
let widgetInstances: Record<string, unknown> = {};

vi.mock('@workbench/WorkbenchContext', () => ({
  useOptionalWorkbenchCommands: () => null,
  useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
    selector({ id: activeProjectId, widgetInstances }),
  useWidgetValuesSelector: (widgetId: string, selector: (values: Record<string, unknown>) => unknown) =>
    selector(
      (Object.values(widgetInstances) as { state?: { values?: Record<string, unknown> }; typeId: string }[]).find(
        (instance) => instance.typeId === widgetId
      )?.state?.values ?? {}
    ),
  useWorkbenchCommands: () => ({
    gallery: { selectItem },
    notifications: { reportError },
    widgets: { patchValues: vi.fn() },
  }),
  useWorkbenchQueries: () => ({
    getProject: (projectId: string) => storedProjects[projectId] ?? null,
    isActiveProject: (projectId: string) => projectId === activeProjectId,
    getSnapshot: () => ({
      activeProject: { id: activeProjectId },
    }),
  }),
}));

const videoItem = { kind: 'video', name: 'clip.mp4' } as unknown as GalleryVideoItem;
const span = { endSeconds: 3, item: videoItem, startSeconds: 2 };

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderAdapter = async (): Promise<void> => {
  await act(() => root.render(<VideoUiAdapterProvider>{null}</VideoUiAdapterProvider>));
};

beforeEach(async () => {
  openResult = { changedWorkspace: false, ok: true, region: 'center', workspaceId: 'compose' };
  activeProjectId = 'project-1';
  open.mockClear();
  selectItem.mockClear();
  revealGalleryItem.mockClear();
  reportError.mockClear();
  storedProjects = {};
  widgetInstances = {};
  galleryQueries.canAddToCachedGalleryBoard.mockReset().mockReturnValue(true);
  galleryQueries.ensureGalleryBoardDestinations.mockReset().mockReturnValue(null);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await renderAdapter();
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();

  const outstanding = getVideoSpanPlaybackRequest();

  if (outstanding) {
    consumeVideoSpanPlaybackRequest(outstanding.token);
  }
});

describe('readValues', () => {
  const videoProject = (values: Record<string, unknown>) => ({
    widgetInstances: { 'video-1': { state: { values }, typeId: 'video' } },
  });

  it("reads the panel's values when called, for drops that commit after the panel changed", () => {
    storedProjects = { 'project-1': videoProject({ positivePrompt: 'before' }) };
    const { readValues } = adapter;

    // No re-render: an async commit holds the adapter it captured, and must still see the store as it is now.
    storedProjects = { 'project-1': videoProject({ positivePrompt: 'after' }) };

    expect(readValues()).toEqual({ positivePrompt: 'after' });
  });

  it('reads the project its writes target, and nothing once that project is gone', async () => {
    storedProjects = { 'project-1': videoProject({ positivePrompt: 'one' }), 'project-2': videoProject({}) };
    const { readValues } = adapter;

    activeProjectId = 'project-2';
    await renderAdapter();

    expect(readValues()).toEqual({ positivePrompt: 'one' });
    storedProjects = {};
    expect(readValues()).toEqual({});
  });
});

describe('getUploadBoardId', () => {
  it('resolves the auto-add board from the ensured listing, and the inbox once the listing refuses it', async () => {
    widgetInstances = {
      'gallery-1': { state: { values: { autoAddBoardId: 'shared', projectBoardId: 'inbox' } }, typeId: 'gallery' },
    };
    await renderAdapter();
    let settleListing!: () => void;
    galleryQueries.ensureGalleryBoardDestinations.mockReturnValueOnce(
      new Promise<void>((resolve) => {
        settleListing = resolve;
      })
    );

    const pending = adapter.getUploadBoardId();
    let resolved: string | null = null;
    void pending.then((boardId) => {
      resolved = boardId;
    });
    await Promise.resolve();

    // Judged only once the listing is in: nothing is trusted unseen.
    expect(resolved).toBeNull();
    settleListing();
    await expect(pending).resolves.toBe('shared');
    expect(galleryQueries.canAddToCachedGalleryBoard).toHaveBeenCalledWith('shared', expect.anything());

    galleryQueries.canAddToCachedGalleryBoard.mockReturnValue(false);
    await expect(adapter.getUploadBoardId()).resolves.toBe('inbox');
  });
});

describe('videoSpanPlayback', () => {
  it("hands the panel the player's report and its changes", () => {
    const listener = vi.fn();
    const unsubscribe = adapter.videoSpanPlayback.subscribe(listener);
    const pause = vi.fn();

    expect(adapter.videoSpanPlayback.getState()).toBeNull();

    publishVideoSpanPlaybackState({ isPlaying: true, pause, token: 41 });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(adapter.videoSpanPlayback.getState()).toMatchObject({ isPlaying: true, token: 41 });

    unsubscribe();
    clearVideoSpanPlaybackState(41);
  });
});

describe('findInGallery', () => {
  it('shows Preview on the stage and then the Gallery, as one gesture', async () => {
    act(() => adapter.findInGallery({ kind: 'video', name: 'clip.mp4' }));

    expect(open.mock.calls).toEqual([
      [[{ surface: 'stage', typeId: 'preview' }, { typeId: 'gallery' }], { focus: 'gallery' }],
    ]);

    await vi.waitFor(() => expect(revealGalleryItem).toHaveBeenCalledTimes(1));

    expect(revealGalleryItem.mock.calls[0]?.[1]).toEqual({ kind: 'video', name: 'clip.mp4' });
  });

  it('describes the press, not the moment its chunk lands', async () => {
    // Claim the reveal ticket before lazy import so later gestures win and the request retains its originating project
    // fence.
    const before = claimGalleryNavigationSequence();

    act(() => adapter.findInGallery({ kind: 'image', name: 'still.png' }));

    const afterPress = claimGalleryNavigationSequence();

    activeProjectId = 'project-2';
    await renderAdapter();

    await vi.waitFor(() => expect(revealGalleryItem).toHaveBeenCalledTimes(1));

    expect(revealGalleryItem.mock.calls[0]?.[2]).toMatchObject({
      accountScope: { epoch: expect.any(Number), signal: expect.any(AbortSignal) },
      projectId: 'project-1',
      sequence: before + 1,
    });
    expect(afterPress).toBe(before + 2);
  });

  it('tells the user when the media cannot be revealed', async () => {
    revealGalleryItem.mockReturnValueOnce(Promise.reject(new Error('Image not found')));

    act(() => adapter.findInGallery({ kind: 'image', name: 'deleted.png' }));

    // Report reveal failure because the widgets have already moved.
    await vi.waitFor(() =>
      expect(reportError).toHaveBeenCalledWith({
        area: 'find-in-gallery',
        message: 'Image not found',
        namespace: 'gallery',
      })
    );
  });
});

describe('playVideoSpanInPreview', () => {
  it('puts the clip in front of the user without disturbing their place in the gallery', () => {
    const revealBefore = getGalleryRevealRequest()?.token;

    let token: number | null = null;
    act(() => {
      token = adapter.playVideoSpanInPreview(span);
    });

    expect(selectItem).toHaveBeenCalledWith(videoItem, 'project-1');
    expect(getVideoSpanPlaybackRequest()).toMatchObject({
      endSeconds: 3,
      itemKey: 'video:clip.mp4',
      startSeconds: 2,
    });
    // The button keeps the token to recognise the player's report on this request.
    expect(token).toBe(getVideoSpanPlaybackRequest()?.token);
    // Auditioning a trim must not scroll the gallery.
    expect(getGalleryRevealRequest()?.token).toBe(revealBefore);
  });

  it('drops a press whose lookup landed after the user switched projects', async () => {
    // A clip resolved after a project switch must not update the previous project's persisted selection.
    const stalePress = adapter.playVideoSpanInPreview;

    activeProjectId = 'project-2';
    await renderAdapter();

    act(() => stalePress(span));

    expect(selectItem).not.toHaveBeenCalled();
    expect(getVideoSpanPlaybackRequest()).toBeNull();
  });
});
