import type { GalleryItemRef, GalleryVideoItem } from '@features/gallery';
import type { VideoUiAdapter } from '@features/video';
import type { OpenWorkbenchWidgetResult } from '@workbench/useOpenWorkbenchWidget';
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
let openResult: OpenWorkbenchWidgetResult;
let activeProjectId: string;

const openWorkbenchWidget = vi.fn(() => openResult);
const selectItem = vi.fn();
const reportError = vi.fn();
const revealGalleryItem = vi.fn(
  (_context: unknown, _ref: GalleryItemRef, _ticket: { projectId: string; sequence: number }) => Promise.resolve()
);
/** The regions the project's single gallery instance occupies. */
let galleryRegions: string[] = ['right'];

vi.mock('@features/video', () => ({
  VideoUiProvider: ({ adapter: next, children }: { adapter: VideoUiAdapter; children: ReactNode }) => {
    adapter = next;
    return children;
  },
}));
vi.mock('@features/gallery/queries', () => ({ invalidateGallery: vi.fn() }));
vi.mock('@workbench/image-actions/revealGalleryItem', () => ({ revealGalleryItem }));
vi.mock('@tanstack/react-query', () => ({ useQueryClient: () => ({}) }));
vi.mock('@workbench/settings/store', () => ({ useWorkbenchPreferenceSelector: () => false }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => openWorkbenchWidget }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
    selector({ id: activeProjectId, widgetInstances: {} }),
  useWorkbenchCommands: () => ({
    gallery: { selectItem },
    notifications: { reportError },
    widgets: { patchValues: vi.fn() },
  }),
  useWorkbenchQueries: () => ({
    isActiveProject: (projectId: string) => projectId === activeProjectId,
    getSnapshot: () => ({
      activeProject: {
        id: activeProjectId,
        widgetInstances: { 'gallery-1': { typeId: 'gallery' } },
        widgetRegions: Object.fromEntries(
          (['left', 'right', 'bottom', 'center'] as const).map((region) => [
            region,
            { instanceIds: galleryRegions.includes(region) ? ['gallery-1'] : [] },
          ])
        ),
      },
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
  openResult = { ok: true, region: 'center' };
  activeProjectId = 'project-1';
  openWorkbenchWidget.mockClear();
  selectItem.mockClear();
  revealGalleryItem.mockClear();
  reportError.mockClear();
  galleryRegions = ['right'];
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
  it('raises the grid where it already lives, not where the manifest would place it', async () => {
    act(() => adapter.findInGallery({ kind: 'video', name: 'clip.mp4' }));

    // Find must relocate Gallery explicitly; an unqualified open can leave it listed in two regions. Raise Preview
    // first so Gallery wins a shared region.
    expect(openWorkbenchWidget.mock.calls).toEqual([
      ['preview', { preferredRegions: ['center'], requireCenterView: true }],
      ['gallery', { preferredRegions: ['right'] }],
    ]);

    await vi.waitFor(() => expect(revealGalleryItem).toHaveBeenCalledTimes(1));

    expect(revealGalleryItem.mock.calls[0]?.[1]).toEqual({ kind: 'video', name: 'clip.mp4' });
  });

  it('falls back to default placement when the project has no grid at all', async () => {
    galleryRegions = [];
    await renderAdapter();

    act(() => adapter.findInGallery({ kind: 'image', name: 'still.png' }));

    expect(openWorkbenchWidget.mock.calls.at(-1)).toEqual(['gallery', undefined]);
  });

  it('describes the press, not the moment its chunk lands', async () => {
    // Claim the reveal ticket before lazy import so later gestures win and the request retains its originating
    // project fence.
    const before = claimGalleryNavigationSequence();

    act(() => adapter.findInGallery({ kind: 'image', name: 'still.png' }));

    const afterPress = claimGalleryNavigationSequence();

    activeProjectId = 'project-2';
    await renderAdapter();

    await vi.waitFor(() => expect(revealGalleryItem).toHaveBeenCalledTimes(1));

    expect(revealGalleryItem.mock.calls[0]?.[2]).toEqual({ projectId: 'project-1', sequence: before + 1 });
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

  it('says nothing about a failed gesture the user has already moved past', async () => {
    revealGalleryItem.mockReturnValueOnce(Promise.reject(new Error('Image not found')));

    act(() => adapter.findInGallery({ kind: 'image', name: 'deleted.png' }));
    // Any later navigation supersedes it — here, a second press.
    act(() => adapter.findInGallery({ kind: 'image', name: 'other.png' }));

    await vi.waitFor(() => expect(revealGalleryItem).toHaveBeenCalledTimes(2));

    expect(reportError).not.toHaveBeenCalled();
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

  it('changes nothing at all when Preview refuses to open', () => {
    openResult = { ok: false, reason: 'unavailable' };

    let token: number | null = 0;
    act(() => {
      token = adapter.playVideoSpanInPreview(span);
    });

    // Failed play must change neither selection nor pending span, which a future player could consume
    // unexpectedly.
    expect(selectItem).not.toHaveBeenCalled();
    expect(getVideoSpanPlaybackRequest()).toBeNull();
    // Nothing was asked, so there is no report to wait for.
    expect(token).toBeNull();
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
