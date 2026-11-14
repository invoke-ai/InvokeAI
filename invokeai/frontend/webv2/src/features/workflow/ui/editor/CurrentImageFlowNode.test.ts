import type { GalleryImageItem, GalleryVideoItem } from '@features/gallery/contracts';

import { createInstance } from 'i18next';
import { describe, expect, it, vi } from 'vitest';

import { createVideoErrorRecovery, resolveCurrentImageExecutionState } from './CurrentImageFlowNode';

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const enResources = Object.values(englishCatalogModules)[0] as Record<string, unknown>;

const SAMPLE_VIDEO: GalleryVideoItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-11-13T00:00:00Z',
  durationSeconds: 4,
  fps: 24,
  fullUrl: '/api/v1/videos/i/clip.mp4/full',
  height: 576,
  isIntermediate: false,
  kind: 'video',
  name: 'clip.mp4',
  starred: false,
  thumbnailUrl: '/api/v1/videos/i/clip.mp4/thumbnail',
  width: 1024,
};

const SAMPLE_IMAGE: GalleryImageItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-11-13T00:00:00Z',
  fullUrl: '/api/v1/images/i/still.png/full',
  height: 1024,
  isIntermediate: false,
  kind: 'image',
  name: 'still.png',
  starred: false,
  thumbnailUrl: '/api/v1/images/i/still.png/thumbnail',
  width: 1024,
};

describe('resolveCurrentImageExecutionState', () => {
  it('keeps selected video playable and draggable when a workflow run is only waiting/pending', () => {
    const state = resolveCurrentImageExecutionState({
      activeTargetCount: 0,
      displayedItem: SAMPLE_VIDEO,
      isWorkflowRunning: true,
      progressImage: null,
      progressPercentage: null,
    });

    expect(state).toEqual({
      dragRef: { kind: 'video', name: 'clip.mp4' },
      hasLiveFrame: false,
      isGenerating: false,
      isShowingVideo: true,
      isWaiting: true,
      progressPercent: null,
    });
  });

  it('reports generating percentage during active execution while keeping media visible until a live frame arrives', () => {
    const beforeFrame = resolveCurrentImageExecutionState({
      activeTargetCount: 1,
      displayedItem: SAMPLE_VIDEO,
      isWorkflowRunning: true,
      progressImage: null,
      progressPercentage: 0.375,
    });

    expect(beforeFrame).toEqual({
      dragRef: { kind: 'video', name: 'clip.mp4' },
      hasLiveFrame: false,
      isGenerating: true,
      isShowingVideo: true,
      isWaiting: false,
      progressPercent: 38,
    });

    const withLiveFrame = resolveCurrentImageExecutionState({
      activeTargetCount: 1,
      displayedItem: SAMPLE_VIDEO,
      isWorkflowRunning: true,
      progressImage: { dataUrl: 'data:image/png;base64,AA==', height: 64, width: 64 },
      progressPercentage: 0.75,
    });

    expect(withLiveFrame).toEqual({
      dragRef: null,
      hasLiveFrame: true,
      isGenerating: true,
      isShowingVideo: false,
      isWaiting: false,
      progressPercent: 75,
    });
  });

  it('exposes image drag ref and idle status when nothing is queued or running', () => {
    const idle = resolveCurrentImageExecutionState({
      activeTargetCount: 0,
      displayedItem: SAMPLE_IMAGE,
      isWorkflowRunning: false,
      progressImage: null,
      progressPercentage: null,
    });

    expect(idle).toEqual({
      dragRef: { kind: 'image', name: 'still.png' },
      hasLiveFrame: false,
      isGenerating: false,
      isShowingVideo: false,
      isWaiting: false,
      progressPercent: null,
    });
  });
});

describe('createVideoErrorRecovery', () => {
  const makeFakeVideo = () => ({ load: vi.fn() }) as unknown as HTMLVideoElement;

  it('refreshes the protected-media cookie once and reloads the video before surfacing a failure', async () => {
    const video = makeFakeVideo();
    let failed = false;
    const refreshCookie = vi.fn<() => Promise<boolean>>().mockResolvedValue(true);

    const recovery = createVideoErrorRecovery({
      getAccountEpoch: () => 3,
      onFailedChange: (next) => {
        failed = next;
      },
      refreshCookie,
    });

    recovery.handleError(video, () => true);
    await vi.waitFor(() => expect(video.load).toHaveBeenCalledTimes(1));

    expect(refreshCookie).toHaveBeenCalledTimes(1);
    expect(failed).toBe(false);

    // A second error after the automatic refresh surfaces the failure UI instead of looping forever.
    recovery.handleError(video, () => true);
    expect(failed).toBe(true);
    expect(refreshCookie).toHaveBeenCalledTimes(1);

    // Manual retry clears failure, reloads, and rearms the one-shot cookie refresh.
    recovery.retry(video);
    expect(failed).toBe(false);
    expect(video.load).toHaveBeenCalledTimes(2);

    recovery.handleError(video, () => true);
    await vi.waitFor(() => expect(refreshCookie).toHaveBeenCalledTimes(2));
  });

  it('surfaces failure immediately when cookie refresh fails and ignores stale account epochs', async () => {
    const video = makeFakeVideo();
    let epoch = 10;
    let failed = false;
    let resolveRefresh!: (ok: boolean) => void;
    const refreshCookie = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          resolveRefresh = resolve;
        })
    );

    const recovery = createVideoErrorRecovery({
      getAccountEpoch: () => epoch,
      onFailedChange: (next) => {
        failed = next;
      },
      refreshCookie,
    });

    // Duplicate error while refresh is pending is ignored.
    recovery.handleError(video, () => true);
    recovery.handleError(video, () => true);
    expect(refreshCookie).toHaveBeenCalledTimes(1);

    // Account changes while refresh is in flight -> neither reload nor mark failed on the new session.
    epoch = 11;
    resolveRefresh(true);
    await Promise.resolve();
    expect(video.load).not.toHaveBeenCalled();
    expect(failed).toBe(false);

    // Reset via manual retry, then fail refresh -> marks failed.
    recovery.retry(video);
    expect(video.load).toHaveBeenCalledTimes(1);
    refreshCookie.mockResolvedValueOnce(false);
    recovery.handleError(video, () => true);
    await vi.waitFor(() => expect(failed).toBe(true));
  });
});

describe('CurrentImageFlowNode locale keys', () => {
  it('resolves every translation key used by CurrentImageFlowNode in en.json', async () => {
    const i18n = createInstance();
    await i18n.init({
      initAsync: false,
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation: enResources } },
    });

    const requiredKeys = [
      'common.copyUnavailable',
      'common.status.waiting',
      'widgets.preview.copyCurrentFrame',
      'widgets.preview.copyCurrentFrameDrawFailed',
      'widgets.preview.copyCurrentFrameEncodeFailed',
      'widgets.preview.copyCurrentFrameNotReady',
      'widgets.preview.copyCurrentFrameStale',
      'widgets.preview.copyCurrentFrameSuccess',
      'widgets.preview.copyCurrentFrameUnsupported',
      'widgets.preview.copyCurrentFrameWriteFailed',
      'widgets.preview.dragVideo',
      'widgets.preview.videoFailed',
      'widgets.preview.videoRetry',
      'widgets.workflow.addNodeDialog.currentImageTitle',
      'widgets.workflow.currentImageDragImage',
      'widgets.workflow.currentImageEmpty',
      'widgets.workflow.currentImageResizeHandle',
      'widgets.workflow.generating',
    ];

    for (const key of requiredKeys) {
      expect(i18n.exists(key), `Missing locale key: ${key}`).toBe(true);
    }
  });
});
