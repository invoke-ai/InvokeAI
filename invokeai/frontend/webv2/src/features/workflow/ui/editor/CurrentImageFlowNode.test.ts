import type { GalleryImageItem, GalleryVideoItem } from '@features/gallery/contracts';
import type { QueueActiveSession } from '@features/queue/contracts';

import { describe, expect, it } from 'vitest';

import {
  resolveCurrentImageActiveSession,
  resolveCurrentImageExecutionState,
  resolveProgressPercentage,
} from './CurrentImageFlowNode';

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

const makeSession = (
  overrides: Partial<QueueActiveSession> & Pick<QueueActiveSession, 'backendItemId' | 'itemIndex' | 'queueItemId'>
): QueueActiveSession => ({
  hasOwnFrame: false,
  height: 512,
  id: `${overrides.queueItemId}:${overrides.itemIndex}`,
  itemCount: 2,
  label: 'Workflow',
  sourceId: 'workflow',
  state: 'running',
  width: 512,
  ...overrides,
});

describe('resolveCurrentImageActiveSession', () => {
  it('follows the newest running framed session stably, prefers running over settling, and tracks unframed starts', () => {
    const slot1 = makeSession({ backendItemId: 101, hasOwnFrame: true, itemIndex: 1, queueItemId: 'batch-1' });
    const slot2 = makeSession({ backendItemId: 102, hasOwnFrame: true, itemIndex: 2, queueItemId: 'batch-1' });

    // Two running framed slots in backend start order: always follow newest (slot 2) without flipping per frame.
    expect(resolveCurrentImageActiveSession([slot1, slot2])).toEqual({ liveSession: slot2, trackedSession: slot2 });

    // Slot 2 finishes (settling) while slot 1 is still running: switch to running slot 1.
    const slot2Settling = { ...slot2, state: 'settling' as const };
    expect(resolveCurrentImageActiveSession([slot1, slot2Settling])).toEqual({
      liveSession: slot1,
      trackedSession: slot1,
    });

    // Only a settling framed slot remains: keep holding its frame until results land.
    expect(resolveCurrentImageActiveSession([slot2Settling])).toEqual({
      liveSession: slot2Settling,
      trackedSession: slot2Settling,
    });

    // Restarted/starting slot with no frame yet: liveSession stays null (no stale frame), trackedSession reports progress.
    const unframed = makeSession({ backendItemId: 101, hasOwnFrame: false, itemIndex: 1, queueItemId: 'run-1' });
    expect(resolveCurrentImageActiveSession([unframed])).toEqual({
      liveSession: null,
      trackedSession: unframed,
    });

    expect(resolveCurrentImageActiveSession([])).toEqual({ liveSession: null, trackedSession: null });
  });
});

describe('resolveProgressPercentage', () => {
  const slot1 = { itemIndex: 1, queueItemId: 'batch-1' };

  it('uses coordinator percentage only when activeItemIndex matches the displayed slot and falls back per-slot otherwise', () => {
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: 1, percentage: 0.5 },
        activeTarget: slot1,
        slotProgressPercentage: null,
      })
    ).toBe(0.5);
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: 2, percentage: 0.2 },
        activeTarget: slot1,
        slotProgressPercentage: 0.75,
      })
    ).toBe(0.75);
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: undefined, percentage: 0.45 },
        activeTarget: slot1,
        slotProgressPercentage: null,
      })
    ).toBeNull();
    expect(
      resolveProgressPercentage({
        activeProgress: { activeItemIndex: undefined, percentage: null },
        activeTarget: slot1,
        slotProgressPercentage: 0.75,
      })
    ).toBe(0.75);
  });
});

describe('resolveCurrentImageExecutionState', () => {
  it('separates Waiting, active pre-frame Generating, live-frame Generating, and Idle states', () => {
    expect(
      resolveCurrentImageExecutionState({
        displayedItem: SAMPLE_VIDEO,
        hasRunningSession: false,
        isWorkflowRunning: true,
        progressImage: null,
        progressPercentage: null,
      })
    ).toEqual({
      dragRef: { kind: 'video', name: 'clip.mp4' },
      hasLiveFrame: false,
      isGenerating: false,
      isShowingVideo: true,
      isWaiting: true,
      progressPercent: null,
    });

    expect(
      resolveCurrentImageExecutionState({
        displayedItem: SAMPLE_VIDEO,
        hasRunningSession: true,
        isWorkflowRunning: true,
        progressImage: null,
        progressPercentage: 0.375,
      })
    ).toEqual({
      dragRef: { kind: 'video', name: 'clip.mp4' },
      hasLiveFrame: false,
      isGenerating: true,
      isShowingVideo: true,
      isWaiting: false,
      progressPercent: 38,
    });

    expect(
      resolveCurrentImageExecutionState({
        displayedItem: SAMPLE_IMAGE,
        hasRunningSession: false,
        isWorkflowRunning: false,
        progressImage: null,
        progressPercentage: null,
      })
    ).toEqual({
      dragRef: { kind: 'image', name: 'still.png' },
      hasLiveFrame: false,
      isGenerating: false,
      isShowingVideo: false,
      isWaiting: false,
      progressPercent: null,
    });
  });
});
