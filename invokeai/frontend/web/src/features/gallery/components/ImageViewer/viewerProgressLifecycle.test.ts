import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { atom, map } from 'nanostores';
import type { S } from 'services/api/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ViewerProgressDataMap, ViewerProgressStores } from './viewerProgressLifecycle';
import { createViewerProgressLifecycle, RESOLVE_TIMEOUT_MS } from './viewerProgressLifecycle';

const buildProgressImage = (itemId: number): ProgressImageType =>
  ({
    dataURL: `data:image/png;base64,item-${itemId}`,
    width: 512,
    height: 512,
  }) as ProgressImageType;

const buildProgressEvent = (overrides: Partial<S['InvocationProgressEvent']> = {}): S['InvocationProgressEvent'] =>
  ({
    queue_id: 'default',
    item_id: 1,
    batch_id: 'batch-1',
    origin: null,
    destination: null,
    user_id: 'user-1',
    session_id: 'session-1',
    invocation_source_id: 'node-1',
    invocation: { id: 'node-1', type: 'test_node' },
    message: 'denoising',
    percentage: 0.5,
    image: null,
    ...overrides,
  }) as S['InvocationProgressEvent'];

const buildTerminalEvent = (
  overrides: Partial<S['QueueItemStatusChangedEvent']> = {}
): S['QueueItemStatusChangedEvent'] =>
  ({
    queue_id: 'default',
    item_id: 1,
    batch_id: 'batch-1',
    origin: null,
    destination: null,
    user_id: 'user-1',
    status: 'completed',
    ...overrides,
  }) as S['QueueItemStatusChangedEvent'];

const buildQueueClearedEvent = (userId: string | null): S['QueueClearedEvent'] =>
  ({ queue_id: 'default', user_id: userId }) as S['QueueClearedEvent'];

describe('viewerProgressLifecycle', () => {
  let stores: ViewerProgressStores;
  let lifecycle: ReturnType<typeof createViewerProgressLifecycle>;

  beforeEach(() => {
    vi.useFakeTimers();
    stores = {
      $progressEvent: atom<S['InvocationProgressEvent'] | null>(null),
      $progressImage: atom<ProgressImageType | null>(null),
      $progressData: map<ViewerProgressDataMap>({}),
      $isProgressImageResolving: atom(false),
      finishedQueueItemIds: new Map<number, boolean>(),
      itemIdBySessionId: new Map<string, number>(),
    };
    lifecycle = createViewerProgressLifecycle(stores);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  const startTwoSessions = () => {
    // B posts a preview first, then A — A owns the shared single-image preview.
    const eventB = buildProgressEvent({ item_id: 2, session_id: 'session-2', image: buildProgressImage(2) });
    const eventA = buildProgressEvent({ item_id: 1, session_id: 'session-1', image: buildProgressImage(1) });
    lifecycle.recordProgress(eventB);
    lifecycle.recordProgress(eventA);
    return { eventA, eventB };
  };

  describe('recordProgress', () => {
    it('tracks per-session data and hands the shared preview to the latest reporter', () => {
      const { eventA } = startTwoSessions();
      expect(stores.$progressEvent.get()).toBe(eventA);
      expect(stores.$progressImage.get()).toBe(eventA.image);
      expect(stores.$progressData.get()[1]?.itemId).toBe(1);
      expect(stores.$progressData.get()[2]?.itemId).toBe(2);
    });

    it('ignores events for finished items so trailing progress cannot repopulate the preview', () => {
      stores.finishedQueueItemIds.set(1, true);
      const handled = lifecycle.recordProgress(buildProgressEvent({ item_id: 1, image: buildProgressImage(1) }));
      expect(handled).toBe(false);
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressData.get()[1]).toBeUndefined();
    });
  });

  describe('onTerminal', () => {
    it.each(['completed', 'canceled', 'failed'] as const)(
      'hands the shared preview to the remaining session when its owner reaches %s',
      (status) => {
        const { eventB } = startTwoSessions();
        lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status }), true);
        // B immediately becomes the single visible progress image and indicator.
        expect(stores.$progressEvent.get()).toBe(eventB);
        expect(stores.$progressImage.get()).toBe(eventB.image);
        expect(stores.$progressData.get()[1]).toBeUndefined();
        expect(stores.$progressData.get()[2]?.itemId).toBe(2);
        // No resolve illusion may be pending — it would swap B's live preview for A's final image.
        expect(stores.$isProgressImageResolving.get()).toBe(false);
        lifecycle.onFinalImageLoaded('session-1');
        expect(stores.$progressEvent.get()).toBe(eventB);
        expect(stores.$progressImage.get()).toBe(eventB.image);
      }
    );

    it('leaves the promoted session up when no further event arrives', () => {
      // The handoff must not be on borrowed time: after A finishes, B may go a long while before
      // its next progress event (a video render's steps are seconds apart), and no timer armed by
      // A's completion may take B's preview down in the meantime.
      const { eventB } = startTwoSessions();
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status: 'completed' }), true);
      lifecycle.onFinalImageLoaded('session-1');
      vi.advanceTimersByTime(RESOLVE_TIMEOUT_MS * 10);
      expect(stores.$progressEvent.get()).toBe(eventB);
      expect(stores.$progressImage.get()).toBe(eventB.image);
      expect(stores.$progressData.get()[2]?.itemId).toBe(2);
    });

    it('hands the shared preview to the remaining session even when auto-switch is off', () => {
      const { eventB } = startTwoSessions();
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status: 'canceled' }), false);
      expect(stores.$progressEvent.get()).toBe(eventB);
      expect(stores.$progressImage.get()).toBe(eventB.image);
    });

    it('promotes the most recently updated remaining session when several remain', () => {
      startTwoSessions();
      const eventC = buildProgressEvent({ item_id: 3, session_id: 'session-3', image: buildProgressImage(3) });
      lifecycle.recordProgress(eventC);
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 3, status: 'canceled' }), true);
      // C owned the preview; A reported more recently than B, so A takes over.
      expect(stores.$progressEvent.get()?.item_id).toBe(1);
    });

    it('leaves the shared preview alone when a non-owner terminates', () => {
      const { eventA } = startTwoSessions();
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 2, status: 'canceled' }), true);
      expect(stores.$progressEvent.get()).toBe(eventA);
      expect(stores.$progressImage.get()).toBe(eventA.image);
      expect(stores.$progressData.get()[2]).toBeUndefined();
    });

    it('clears immediately when the last session is canceled', () => {
      const eventA = buildProgressEvent({ item_id: 1, image: buildProgressImage(1) });
      lifecycle.recordProgress(eventA);
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status: 'canceled' }), true);
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressImage.get()).toBeNull();
      expect(stores.$isProgressImageResolving.get()).toBe(false);
    });

    it('runs the resolve illusion when the last session completes with auto-switch on', () => {
      const eventA = buildProgressEvent({ item_id: 1, image: buildProgressImage(1) });
      lifecycle.recordProgress(eventA);
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status: 'completed' }), true);
      // The preview is retained until the final image loads, "resolving" into it.
      expect(stores.$progressEvent.get()).toBe(eventA);
      expect(stores.$isProgressImageResolving.get()).toBe(true);
      lifecycle.onFinalImageLoaded('session-1');
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressImage.get()).toBeNull();
      expect(stores.$isProgressImageResolving.get()).toBe(false);
    });

    it('runs no resolve illusion when the completed item never reported progress', () => {
      // Nothing is retained, so arming the illusion would only leave the resolving flag stuck on.
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status: 'completed' }), true);
      expect(stores.$isProgressImageResolving.get()).toBe(false);
      expect(stores.$progressEvent.get()).toBeNull();
    });

    it('ignores repeat terminal events for an already-finished item', () => {
      const { eventA } = startTwoSessions();
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 2, status: 'canceled' }), true);
      expect(lifecycle.onTerminal(buildTerminalEvent({ item_id: 2, status: 'canceled' }), true)).toBe(false);
      expect(stores.$progressEvent.get()).toBe(eventA);
    });
  });

  describe('onFinalImageLoaded', () => {
    it('ignores a late load from a session that finished before the current preview owner', () => {
      const { eventB } = startTwoSessions();
      // A completes first and hands the shared preview to the still-running B...
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status: 'completed' }), true);
      // ...then B completes and starts its own resolve illusion, retaining B's last frame.
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 2, status: 'completed' }), true);
      expect(stores.$isProgressImageResolving.get()).toBe(true);
      // A's final image only now finishes loading. Clearing here would cut B's illusion short.
      lifecycle.onFinalImageLoaded('session-1');
      expect(stores.$progressEvent.get()).toBe(eventB);
      expect(stores.$progressImage.get()).toBe(eventB.image);
      expect(stores.$isProgressImageResolving.get()).toBe(true);
      // B's own final image ends the illusion.
      lifecycle.onFinalImageLoaded('session-2');
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressImage.get()).toBeNull();
      expect(stores.$isProgressImageResolving.get()).toBe(false);
    });

    it('clears the retained preview on a timeout when its final image never loads', () => {
      // The retained session's image may never load: the load can fail (the viewer's <Image>
      // reports errors through onError, not onLoad), or auto-switch may end up selecting a
      // concurrently-completed session's image, so the retained session's image is never
      // rendered. An ignored load may have been the last one coming, so the preview — an opaque
      // overlay — must not be left covering the viewer until the next generation.
      const { eventB } = startTwoSessions();
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status: 'completed' }), true);
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 2, status: 'completed' }), true);
      lifecycle.onFinalImageLoaded('session-1');
      expect(stores.$progressEvent.get()).toBe(eventB);
      vi.advanceTimersByTime(RESOLVE_TIMEOUT_MS);
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressImage.get()).toBeNull();
      expect(stores.$isProgressImageResolving.get()).toBe(false);
    });

    it.each([
      [
        'a still-running session takes the preview over',
        () => {
          lifecycle.recordProgress(
            buildProgressEvent({ item_id: 2, session_id: 'session-2', image: buildProgressImage(2) })
          );
        },
      ],
      [
        'the final image loads first',
        () => {
          lifecycle.onFinalImageLoaded('session-1');
        },
      ],
      [
        'the preview state is reset',
        () => {
          lifecycle.reset();
        },
      ],
    ])('disarms the resolve timeout when %s', (_desc, takeOver) => {
      const eventA = buildProgressEvent({ item_id: 1, image: buildProgressImage(1) });
      lifecycle.recordProgress(eventA);
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status: 'completed' }), true);
      expect(vi.getTimerCount()).toBe(1);
      takeOver();
      // The timeout must never fire against a preview that has since been replaced or dropped.
      // Asserting the timer is gone, not just that the state survives it, is what makes this bite
      // for the takeovers that leave the stores null — there, a leaked timer would clear state
      // that is already clear.
      expect(vi.getTimerCount()).toBe(0);
      const eventAfterTakeOver = stores.$progressEvent.get();
      vi.advanceTimersByTime(RESOLVE_TIMEOUT_MS * 2);
      expect(stores.$progressEvent.get()).toBe(eventAfterTakeOver);
    });

    it.each([
      ['an image with no session (e.g. an upload)', null],
      ['an image from a session this viewer never tracked', 'session-from-a-previous-visit'],
    ])('still clears the retained preview on %s', (_desc, sessionId) => {
      const eventA = buildProgressEvent({ item_id: 1, image: buildProgressImage(1) });
      lifecycle.recordProgress(eventA);
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 1, status: 'completed' }), true);
      expect(stores.$isProgressImageResolving.get()).toBe(true);
      // Unattributable loads keep the safety net: a retained preview must not cover the viewer
      // indefinitely just because the completed item's own image never loads.
      lifecycle.onFinalImageLoaded(sessionId);
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressImage.get()).toBeNull();
      expect(stores.$isProgressImageResolving.get()).toBe(false);
    });

    it('clears on a load from a session tracked before a reset', () => {
      // Attributions are dropped along with the state they describe, so images generated before a
      // disconnect or socket swap can still end a later session's illusion.
      startTwoSessions();
      lifecycle.reset();
      const eventC = buildProgressEvent({ item_id: 3, session_id: 'session-3', image: buildProgressImage(3) });
      lifecycle.recordProgress(eventC);
      lifecycle.onTerminal(buildTerminalEvent({ item_id: 3, status: 'completed' }), true);
      expect(stores.$isProgressImageResolving.get()).toBe(true);
      lifecycle.onFinalImageLoaded('session-1');
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$isProgressImageResolving.get()).toBe(false);
    });
  });

  describe('onQueueCleared', () => {
    it.each([
      ['an unscoped (admin or single-user) clear', null, 'user-1'],
      ["the current user's scoped clear", 'user-1', 'user-1'],
    ])('drops all previews and blocks trailing progress on %s', (_desc, clearedUserId, currentUserId) => {
      startTwoSessions();
      const applied = lifecycle.onQueueCleared(buildQueueClearedEvent(clearedUserId), currentUserId);
      expect(applied).toBe(true);
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressImage.get()).toBeNull();
      expect(stores.$progressData.get()).toEqual({});
      // A worker claimed between the clear's cancellation pass and deletion is stopped only by
      // this event — its trailing progress must not repopulate the preview.
      expect(lifecycle.recordProgress(buildProgressEvent({ item_id: 1, image: buildProgressImage(1) }))).toBe(false);
      expect(lifecycle.recordProgress(buildProgressEvent({ item_id: 2, image: buildProgressImage(2) }))).toBe(false);
    });

    it('blocks trailing progress from an item claimed after the clear cancelled the running ones', () => {
      // The clear cancels the items already running before deleting the rows, so a worker that
      // claims an item in between never gets a terminal event: only its in_progress claim, which
      // precedes the deletion, tells the viewer the item exists. Its first progress event arrives
      // seconds later — a preview for a deleted item that nothing would ever take down.
      expect(lifecycle.onItemStarted(7)).toBe(true);
      expect(lifecycle.onQueueCleared(buildQueueClearedEvent(null), 'user-1')).toBe(true);
      expect(
        lifecycle.recordProgress(
          buildProgressEvent({ item_id: 7, session_id: 'session-7', image: buildProgressImage(7) })
        )
      ).toBe(false);
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressImage.get()).toBeNull();
    });

    it('blocks trailing progress from a session that had not produced a preview image yet', () => {
      // Item 1 has only reported progress without an image, so it is absent from $progressData and
      // does not own the shared globals once item 2 reports an image.
      lifecycle.recordProgress(buildProgressEvent({ item_id: 1, session_id: 'session-1', image: null }));
      lifecycle.recordProgress(
        buildProgressEvent({ item_id: 2, session_id: 'session-2', image: buildProgressImage(2) })
      );
      expect(lifecycle.onQueueCleared(buildQueueClearedEvent(null), 'user-1')).toBe(true);
      // Its first image event must not resurrect the preview the clear just dropped.
      expect(lifecycle.recordProgress(buildProgressEvent({ item_id: 1, image: buildProgressImage(1) }))).toBe(false);
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressImage.get()).toBeNull();
      expect(stores.$progressData.get()).toEqual({});
    });

    it.each([
      ["another user's scoped clear", 'user-2'],
      ['the sanitized broadcast of a foreign scoped clear', 'redacted'],
    ])('leaves previews alone on %s', (_desc, clearedUserId) => {
      const { eventA } = startTwoSessions();
      const applied = lifecycle.onQueueCleared(buildQueueClearedEvent(clearedUserId), 'user-1');
      expect(applied).toBe(false);
      expect(stores.$progressEvent.get()).toBe(eventA);
      expect(stores.$progressData.get()[1]?.itemId).toBe(1);
      expect(stores.$progressData.get()[2]?.itemId).toBe(2);
    });
  });

  describe('reset', () => {
    it('drops all preview state without marking items finished', () => {
      startTwoSessions();
      stores.$isProgressImageResolving.set(true);
      lifecycle.reset();
      expect(stores.$progressEvent.get()).toBeNull();
      expect(stores.$progressImage.get()).toBeNull();
      expect(stores.$progressData.get()).toEqual({});
      expect(stores.$isProgressImageResolving.get()).toBe(false);
      // A new connection's events for a re-used id are not blocked — reset is not a cancel.
      expect(lifecycle.recordProgress(buildProgressEvent({ item_id: 1, image: buildProgressImage(1) }))).toBe(true);
    });
  });
});
