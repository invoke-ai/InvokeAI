import type { GalleryImageItem, GalleryVideoItem } from '@features/gallery/contracts';

import { createInstance } from 'i18next';
import { describe, expect, it, vi } from 'vitest';

import {
  createVideoErrorRecovery,
  resolveActiveBackendItemId,
  resolveCurrentImageExecutionState,
  resolveLiveProgressImage,
  resolveProgressPercentage,
  resolveProgressTarget,
} from './CurrentImageFlowNode';

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

describe('resolveProgressTarget', () => {
  it('selects the displayed progress frame target when concurrent runs are active', () => {
    const activeTargets = [
      { itemIndex: 1, queueItemId: 'run-a' },
      { itemIndex: 1, queueItemId: 'run-b' },
    ];
    const frameFromRunB = {
      dataUrl: 'data:image/png;base64,AA==',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'run-b' },
      width: 64,
    };

    expect(resolveProgressTarget(frameFromRunB, activeTargets)).toEqual({ itemIndex: 1, queueItemId: 'run-b' });
  });

  it('falls back to the oldest active target before a frame arrives or when the frame has no target', () => {
    const activeTargets = [
      { itemIndex: 1, queueItemId: 'run-a' },
      { itemIndex: 1, queueItemId: 'run-b' },
    ];

    expect(resolveProgressTarget(null, activeTargets)).toEqual({ itemIndex: 1, queueItemId: 'run-a' });
    expect(
      resolveProgressTarget({ dataUrl: 'data:image/png;base64,AA==', height: 64, width: 64 }, activeTargets)
    ).toEqual({ itemIndex: 1, queueItemId: 'run-a' });
    expect(resolveProgressTarget(null, [])).toBeNull();
  });

  it('ignores active targets and frames belonging to another project and pairs batch slot percentages', () => {
    const activeTargets = [
      { itemIndex: 1, queueItemId: 'run-project-a' },
      { itemIndex: 2, queueItemId: 'run-project-b' },
    ];
    const projectBIds = new Set(['run-project-b']);
    const frameFromProjectA = {
      dataUrl: 'data:image/png;base64,AA==',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'run-project-a' },
      width: 64,
    };

    const resolvedTarget = resolveProgressTarget(frameFromProjectA, activeTargets, projectBIds);
    expect(resolvedTarget).toEqual({
      itemIndex: 2,
      queueItemId: 'run-project-b',
    });
    expect(resolveProgressTarget(frameFromProjectA, [activeTargets[0]!], projectBIds)).toBeNull();
    // A retained frame whose slot is no longer running/settling is ignored when followedTargets is empty.
    expect(
      resolveProgressTarget(
        {
          dataUrl: 'data:image/png;base64,AA==',
          height: 64,
          target: { itemIndex: 2, queueItemId: 'run-project-b' },
          width: 64,
        },
        [],
        projectBIds
      )
    ).toBeNull();

    const itemsById = new Map([['run-project-b', { backendItemIds: [101, 102], id: 'run-project-b' }]]);
    expect(resolveActiveBackendItemId(resolvedTarget, itemsById)).toBe(102);
    expect(resolveActiveBackendItemId({ itemIndex: 1, queueItemId: 'run-project-b' }, itemsById)).toBe(101);
    expect(resolveActiveBackendItemId({ itemIndex: 0, queueItemId: 'run-project-b' }, itemsById)).toBeNull();

    const slot1Target = { itemIndex: 1, queueItemId: 'batch-1' };
    // Coordinator active on slot 2 while displaying slot 1 -> use slot 1's per-slot lookup (null if none, 0.3 if set).
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: 2, percentage: 0.2 },
        activeTarget: slot1Target,
      })
    ).toBeNull();
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: 2, percentage: 0.2 },
        activeTarget: slot1Target,
        slotProgressPercentage: 0.75,
      })
    ).toBe(0.75);
    // Coordinator active on slot 1 (including Saved Workflow child calls where parent slotProgressPercentage is null) -> prefer coordinator %.
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: 1, percentage: 0.5 },
        activeTarget: slot1Target,
        slotProgressPercentage: null,
      })
    ).toBe(0.5);
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: 1, percentage: 0.6 },
        activeTarget: slot1Target,
      })
    ).toBe(0.6);
    // Coordinator activeItemIndex cleared when another batch slot finished -> fall back to per-slot percentage instead of null.
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: undefined, percentage: null },
        activeTarget: slot1Target,
        slotProgressPercentage: 0.75,
      })
    ).toBe(0.75);
    // Coordinator activeItemIndex undefined with no per-slot percentage -> do not leak an unattributed percentage onto a concrete batch slot.
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: undefined, percentage: 0.45 },
        activeTarget: slot1Target,
        slotProgressPercentage: null,
      })
    ).toBeNull();
    // Unindexed targets (itemIndex 0) still use the coordinator percentage.
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: undefined, percentage: 0.45 },
        activeTarget: { itemIndex: 0, queueItemId: 'batch-1' },
        slotProgressPercentage: null,
      })
    ).toBe(0.45);
  });

  it('keeps the preferred local running framed slot across foreign updates, and prefers live framed slots over unframed or settling ones', () => {
    const projectATarget1 = { itemIndex: 1, queueItemId: 'run-a-1' };
    const projectATarget2 = { itemIndex: 2, queueItemId: 'run-a-1' };
    const projectAIds = new Set(['run-a-1']);
    const frameFromProjectB = {
      dataUrl: 'data:image/png;base64,BB==',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'run-b-1' },
      width: 64,
    };

    // Even when slot 2 was marked framed after slot 1, keep preferred running slot 1 while only Project B updated.
    expect(
      resolveProgressTarget(
        frameFromProjectB,
        [projectATarget1, projectATarget2],
        projectAIds,
        [projectATarget1, projectATarget2],
        projectATarget1,
        [projectATarget1, projectATarget2]
      )
    ).toEqual(projectATarget1);

    // If preferred slot 1 is settling while slot 2 is still running and framed, switch to running slot 2.
    expect(
      resolveProgressTarget(
        frameFromProjectB,
        [projectATarget2, projectATarget1],
        projectAIds,
        [projectATarget1, projectATarget2],
        projectATarget1,
        [projectATarget2]
      )
    ).toEqual(projectATarget2);

    // If slot 1 restarted after requeue (followed but not yet framed) and global progressImage still carries slot 1's old frame,
    // pick the sibling slot 2 that is actually framed in this project rather than blanking the monitor.
    expect(
      resolveProgressTarget(
        { dataUrl: 'data:image/png;base64,OLD=', height: 64, target: projectATarget1, width: 64 },
        [projectATarget1, projectATarget2],
        projectAIds,
        [projectATarget2],
        projectATarget1,
        [projectATarget1, projectATarget2]
      )
    ).toEqual(projectATarget2);

    // Without a preferred framed slot, fall back to the most recently framed slot in the project.
    expect(
      resolveProgressTarget(frameFromProjectB, [projectATarget1, projectATarget2], projectAIds, [projectATarget2], null)
    ).toEqual(projectATarget2);

    // When no local slot is framed yet, keep the preferred followed local slot over an earlier unframed sibling.
    expect(
      resolveProgressTarget(frameFromProjectB, [projectATarget1, projectATarget2], projectAIds, [], projectATarget2)
    ).toEqual(projectATarget2);

    // A finished/settling slot that sent the latest global frame must not hide a sibling slot that is still running and framed.
    expect(
      resolveProgressTarget(
        { dataUrl: 'data:image/png;base64,DONE=', height: 64, target: projectATarget1, width: 64 },
        [projectATarget2, projectATarget1],
        projectAIds,
        [projectATarget1, projectATarget2],
        projectATarget1,
        [projectATarget2]
      )
    ).toEqual(projectATarget2);

    // Even when no slot is framed yet, a settling preferred slot must not block a still-running sibling slot.
    expect(
      resolveProgressTarget(null, [projectATarget2, projectATarget1], projectAIds, [], projectATarget1, [
        projectATarget2,
      ])
    ).toEqual(projectATarget2);
  });
});

describe('resolveLiveProgressImage', () => {
  it('retains this project slot frame when framed, hides stale frames before a restarted attempt frames, and drops frames when not followed', () => {
    const slotFrame = { dataUrl: 'data:image/png;base64,AA==', height: 64, width: 64 };
    const otherProjectGlobalFrame = {
      dataUrl: 'data:image/png;base64,BB==',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'run-b' },
      width: 64,
    };
    const projectATarget = { itemIndex: 1, queueItemId: 'run-a' };

    // Project A keeps its retained slot frame even though Project B published the latest global frame.
    expect(
      resolveLiveProgressImage({
        activeTarget: projectATarget,
        followedTargets: [projectATarget],
        framedTargets: [projectATarget],
        progressImage: otherProjectGlobalFrame,
        slotProgressImage: slotFrame,
      })
    ).toBe(slotFrame);

    // After requeue + restart before a new frame arrives: slot is followed again, progressImageStore still holds the old frame,
    // but framedTargets does not include the slot -> hide the stale previous-attempt frame.
    expect(
      resolveLiveProgressImage({
        activeTarget: projectATarget,
        followedTargets: [projectATarget],
        framedTargets: [],
        progressImage: { ...slotFrame, target: projectATarget },
        slotProgressImage: slotFrame,
      })
    ).toBeNull();

    // Once the job moves back to pending (no activeTarget), any retained frame is ignored so Waiting returns.
    expect(
      resolveLiveProgressImage({
        activeTarget: null,
        followedTargets: [],
        framedTargets: [],
        progressImage: { ...slotFrame, target: projectATarget },
        slotProgressImage: slotFrame,
      })
    ).toBeNull();

    // Untargeted global frames are rejected when projectQueueItemIds is known, and only accepted in fallback mode.
    expect(
      resolveLiveProgressImage({
        activeTarget: projectATarget,
        allowUntargetedFrame: false,
        followedTargets: [projectATarget],
        framedTargets: [projectATarget],
        progressImage: slotFrame,
        slotProgressImage: null,
      })
    ).toBeNull();
    // Untargeted global frames without an activeTarget are rejected when queueItems are known (allowUntargetedFrame: false),
    // and only accepted in the legacy/test fallback mode without queueItems.
    expect(
      resolveLiveProgressImage({
        activeTarget: null,
        allowUntargetedFrame: false,
        followedTargets: [],
        framedTargets: [],
        progressImage: slotFrame,
        slotProgressImage: null,
      })
    ).toBeNull();
    expect(
      resolveLiveProgressImage({
        activeTarget: null,
        allowUntargetedFrame: true,
        followedTargets: [],
        framedTargets: [],
        progressImage: slotFrame,
        slotProgressImage: null,
      })
    ).toBe(slotFrame);
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

    // A second error after the automatic refresh surfaces the failure UI instead of looping forever, even after partial loads.
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
      'widgets.preview.imageActions',
      'widgets.preview.videoFailed',
      'widgets.preview.videoRetry',
      'widgets.workflow.addNodeDialog.currentImageTitle',
      'widgets.workflow.currentImageEmpty',
      'widgets.workflow.currentImageResetSize',
      'widgets.workflow.currentImageResizeHandle',
      'widgets.workflow.generating',
    ];

    for (const key of requiredKeys) {
      expect(i18n.exists(key), `Missing locale key: ${key}`).toBe(true);
    }
    const resizeInstructions = i18n.t('widgets.workflow.currentImageResizeHandle', { height: 288, width: 320 });
    for (const key of ['Enter', 'Space', 'Home']) {
      expect(resizeInstructions).toContain(key);
    }
  });
});
