/* eslint-disable react/refs */
import { useDndMonitor, type DndMonitorListener, type UniqueIdentifier } from '@dnd-kit/core';
import {
  useCallback,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from 'react';

import type { PreviewNeighbors } from './usePreviewNavigation';

import {
  classifySwipeIntent,
  easeOutQuad,
  getRevealedDirection,
  getSwipeVelocity,
  recordSwipeSample,
  resolveSwipeRelease,
  rubberBand,
  unrubberBand,
  type SwipeDirection,
  type SwipeSample,
} from './previewSwipe';

/**
 * Touch carousel over the fitted preview: one finger drags the media and its neighbors sideways, and release either
 * settles back or slides the neighbor in and navigates. Offsets are written to the elements' `translate` property
 * outside React, leaving `transform` to the loupe. A committed slide parks on the neighbor until the displayed source
 * changes; React itself clears the offset in the commit that renders the new source (see `restStyle`), so the
 * newly selected image lands where its neighbor stood, with no frame of either image out of place.
 */

/** How long a committed slide may wait, fetch included, for its selection to render before returning. */
const PARK_TIMEOUT_MS = 1500;
/** Return time when the gesture itself carries no speed: a cancelled touch or a step that did not happen. */
const RETURN_MS = 220;

type TrackRole = 'content' | 'next' | 'previous';

interface SwipeGesture {
  /** Finger-space offset the gesture started from: nonzero when it caught a returning image. */
  baseOffset: number;
  phase: 'declined' | 'pending' | 'tracking';
  pointerId: number;
  samples: SwipeSample[];
  startX: number;
  startY: number;
  /** Stage width, measured once when the gesture starts tracking. */
  width: number;
}

interface SwipeSettle {
  frame: number;
  onDone: () => void;
  target: number;
}

export interface PreviewSwipeNavigation {
  neighbors: PreviewNeighbors;
  /** Resolves true once the step is dispatched; false leaves the current item in place. */
  onNavigate: (direction: SwipeDirection) => Promise<boolean>;
}

/** Two spellings of "no offset", so each landing changes the rendered style and React writes it. */
const REST_STYLES = { '': { translate: '' }, none: { translate: 'none' } } as const satisfies Record<
  string,
  CSSProperties
>;

const matchesMedia = (query: string): boolean =>
  typeof globalThis.matchMedia === 'function' && globalThis.matchMedia(query).matches;

export const usePreviewSwipe = ({
  displayedSourceToken,
  dragId,
  enabled,
  isZoomed,
  navigation,
}: {
  /** Identity of the displayed media; a change is a selection landing. */
  displayedSourceToken: string | null;
  /** The frame's draggable: once its drag claims a touch, the touch is never a swipe. */
  dragId: UniqueIdentifier;
  enabled: boolean;
  isZoomed: boolean;
  navigation: PreviewSwipeNavigation | null;
}) => {
  const stageRef = useRef<HTMLElement | null>(null);
  const trackRefs = useRef(new Map<TrackRole, HTMLElement>());
  const offsetRef = useRef(0);
  const gestureRef = useRef<SwipeGesture | null>(null);
  const settleRef = useRef<SwipeSettle | null>(null);
  /** Set while a committed slide waits for its selection; the token fences a late navigation result. */
  const parkedRef = useRef<{ timeout: number; token: number } | null>(null);
  const parkSequenceRef = useRef(0);
  // Neighbors mount up front where touch is the primary input, otherwise on the first touch, so mouse sessions (and
  // touchscreen laptops driven by a mouse) never hold or decode them.
  const [showsNeighbors, setShowsNeighbors] = useState(() => matchesMedia('(pointer: coarse)'));
  // Render-time mirrors, so the stable handlers below read the current props.
  const inputRef = useRef({ enabled, isZoomed, navigation });

  inputRef.current = { enabled, isZoomed, navigation };

  const writeOffset = useCallback((offset: number) => {
    offsetRef.current = offset;
    const translate = offset === 0 ? '' : `${offset}px 0`;

    for (const element of trackRefs.current.values()) {
      element.style.translate = translate;
    }
  }, []);

  const canNavigate = useCallback((direction: SwipeDirection): boolean => {
    const neighbors = inputRef.current.navigation?.neighbors;

    return neighbors !== undefined && (direction === 1 ? neighbors.next : neighbors.previous) !== null;
  }, []);

  const stopSettle = useCallback(() => {
    const settle = settleRef.current;

    if (settle) {
      cancelAnimationFrame(settle.frame);
      settleRef.current = null;
    }
  }, []);

  const clearPark = useCallback(() => {
    const parked = parkedRef.current;

    if (parked) {
      window.clearTimeout(parked.timeout);
      parkedRef.current = null;
    }
  }, []);

  const settleTo = useCallback(
    (target: number, durationMs: number, onDone: () => void = () => {}) => {
      stopSettle();
      const from = offsetRef.current;

      if (durationMs <= 0 || from === target || matchesMedia('(prefers-reduced-motion: reduce)')) {
        writeOffset(target);
        onDone();
        return;
      }

      const start = performance.now();
      const tick = (now: number): void => {
        const progress = Math.min(1, (now - start) / durationMs);

        writeOffset(from + (target - from) * easeOutQuad(progress));

        if (progress < 1) {
          settleRef.current!.frame = requestAnimationFrame(tick);
          return;
        }

        settleRef.current = null;
        onDone();
      };

      settleRef.current = { frame: requestAnimationFrame(tick), onDone, target };
    },
    [stopSettle, writeOffset]
  );

  /** Return to rest from wherever the track is. The track has one writer: any live gesture gives it up. */
  const returnToRest = useCallback(
    (durationMs: number) => {
      clearPark();

      if (gestureRef.current) {
        gestureRef.current.phase = 'declined';
      }

      settleTo(0, durationMs);
    },
    [clearPark, settleTo]
  );

  const commit = useCallback(
    (direction: SwipeDirection) => {
      const token = ++parkSequenceRef.current;
      // Bounded from the start, page fetch included: a slow or failed step must never strand the stage on a neighbor.
      const timeout = window.setTimeout(() => {
        if (parkedRef.current?.token === token) {
          returnToRest(RETURN_MS);
        }
      }, PARK_TIMEOUT_MS);

      parkedRef.current = { timeout, token };

      let step: Promise<boolean>;

      try {
        step = inputRef.current.navigation?.onNavigate(direction) ?? Promise.resolve(false);
      } catch {
        step = Promise.resolve(false);
      }

      void step
        .catch(() => false)
        .then((stepped) => {
          // A step that happened parks until its source lands (or times out); a landing already unparked this one.
          if (!stepped && parkedRef.current?.token === token) {
            returnToRest(RETURN_MS);
          }
        });
    },
    [returnToRest]
  );

  const release = useCallback(
    (gesture: SwipeGesture, now: number) => {
      const decision = resolveSwipeRelease({
        canNavigate,
        offset: offsetRef.current,
        velocity: getSwipeVelocity(gesture.samples, now),
        width: gesture.width,
      });

      if (decision.kind === 'cancel') {
        settleTo(0, decision.durationMs);
        return;
      }

      settleTo(-decision.direction * gesture.width, decision.durationMs, () => commit(decision.direction));
    },
    [canNavigate, commit, settleTo]
  );

  /** A touch that never became a swipe leaves a caught image where it was; send it home. */
  const restoreIfStranded = useCallback(() => {
    if (offsetRef.current !== 0 && !parkedRef.current && !settleRef.current) {
      settleTo(0, RETURN_MS);
    }
  }, [settleTo]);

  const handlePointerMove = useCallback(
    (event: PointerEvent) => {
      const gesture = gestureRef.current;

      if (!gesture || gesture.pointerId !== event.pointerId || gesture.phase === 'declined') {
        return;
      }

      const dx = event.clientX - gesture.startX;

      if (gesture.phase === 'pending') {
        const intent = classifySwipeIntent(dx, event.clientY - gesture.startY);

        if (intent === 'pending') {
          return;
        }

        if (intent === 'reject') {
          gesture.phase = 'declined';
          restoreIfStranded();
          return;
        }

        gesture.phase = 'tracking';
        gesture.width = stageRef.current?.clientWidth ?? 0;
      }

      recordSwipeSample(gesture.samples, { time: event.timeStamp, x: event.clientX });

      // A parked track still shows the committed neighbor; the gesture resumes once the new image lands.
      if (parkedRef.current) {
        return;
      }

      const raw = gesture.baseOffset + dx;
      const direction = getRevealedDirection(raw);

      writeOffset(direction !== 0 && !canNavigate(direction) ? rubberBand(raw, gesture.width) : raw);
    },
    [canNavigate, restoreIfStranded, writeOffset]
  );

  const handlePointerEnd = useCallback(
    (event: PointerEvent) => {
      const gesture = gestureRef.current;

      if (!gesture || gesture.pointerId !== event.pointerId) {
        return;
      }

      gestureRef.current = null;

      if (parkedRef.current) {
        return;
      }

      if (gesture.phase !== 'tracking') {
        restoreIfStranded();
        return;
      }

      // A cancel means the browser or a native control took the touch over mid-swipe, not that the user changed their
      // mind: decide it like a lift at the last move, so a swipe that was clearly going somewhere still gets there.
      release(
        gesture,
        event.type === 'pointercancel' ? (gesture.samples.at(-1)?.time ?? event.timeStamp) : event.timeStamp
      );
    },
    [release, restoreIfStranded]
  );

  // Document-level so a finger that leaves the stage still moves and releases the track.
  const stageRefCallback = useCallback(
    (node: HTMLElement | null) => {
      stageRef.current = node;

      if (!node) {
        return;
      }

      const ownerDocument = node.ownerDocument;

      ownerDocument.addEventListener('pointermove', handlePointerMove, { passive: true });
      ownerDocument.addEventListener('pointerup', handlePointerEnd);
      ownerDocument.addEventListener('pointercancel', handlePointerEnd);

      return () => {
        ownerDocument.removeEventListener('pointermove', handlePointerMove);
        ownerDocument.removeEventListener('pointerup', handlePointerEnd);
        ownerDocument.removeEventListener('pointercancel', handlePointerEnd);
        gestureRef.current = null;
        stopSettle();
        clearPark();
      };
    },
    [clearPark, handlePointerEnd, handlePointerMove, stopSettle]
  );

  // Elements that move with the track, registered through stable per-role callbacks so React never re-attaches them.
  const [trackRefCallbacks] = useState(() => {
    const makeTrackRef = (role: TrackRole) => (element: HTMLElement | null) => {
      if (!element) {
        trackRefs.current.delete(role);
        return;
      }

      trackRefs.current.set(role, element);
      element.style.translate = offsetRef.current === 0 ? '' : `${offsetRef.current}px 0`;
    };

    return { content: makeTrackRef('content'), next: makeTrackRef('next'), previous: makeTrackRef('previous') };
  });

  // A new source is a landing (or an outside selection). React owns the visible reset: every track element renders
  // `restStyle`, and flipping it between two spellings of "no offset" makes React write it in the very commit that
  // renders the new source, before paint. The imperative state behind it is retired here, during render, the same way
  // the loupe resets its zoom; that relies on selection arriving through the synchronous project store, so this render
  // is never discarded by a transition or suspense.
  const [rest, setRest] = useState({ token: displayedSourceToken, translate: '' as '' | 'none' });

  if (rest.token !== displayedSourceToken) {
    setRest({ token: displayedSourceToken, translate: rest.translate === '' ? 'none' : '' });

    const wasParked = parkedRef.current !== null;
    const gesture = gestureRef.current;

    clearPark();
    stopSettle();
    offsetRef.current = 0;

    // A finger that flicked again during the slide keeps going on the new image, restarting from rest with its
    // travel since the slide landed; any other gesture belonged to the old image.
    if (gesture && wasParked && gesture.phase === 'tracking') {
      gesture.baseOffset = 0;
      gesture.startX = gesture.samples.at(-1)?.x ?? gesture.startX;
    } else if (gesture && !(wasParked && gesture.phase === 'pending')) {
      gestureRef.current = null;
    }
  }

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLElement>): void => {
      if (event.pointerType !== 'touch') {
        return;
      }

      if (!event.isPrimary) {
        // A second finger is a pinch: hand the stage to the loupe with the image back at rest.
        if (gestureRef.current) {
          gestureRef.current = null;

          if (!parkedRef.current) {
            settleTo(0, 0);
          }
        }

        return;
      }

      setShowsNeighbors(true);

      const { enabled: isEnabled, isZoomed: isZoomedNow, navigation: currentNavigation } = inputRef.current;

      if (!isEnabled || isZoomedNow || !currentNavigation) {
        return;
      }

      const settle = settleRef.current;

      // A committed slide completes now, so a quick follow-up flick is never dropped behind an animation. A return is
      // caught where it is and resumes from the finger travel that put it there; until the touch moves far enough
      // to be a swipe, it can still be a tap or a hold-to-drag, and either sends the image home.
      let baseOffset = 0;

      if (settle && settle.target !== 0) {
        stopSettle();
        writeOffset(settle.target);
        settle.onDone();
      } else if (settle) {
        stopSettle();
        const width = stageRef.current?.clientWidth ?? 0;
        const direction = getRevealedDirection(offsetRef.current);

        baseOffset =
          direction !== 0 && !canNavigate(direction) ? unrubberBand(offsetRef.current, width) : offsetRef.current;
      }

      gestureRef.current = {
        baseOffset,
        phase: 'pending',
        pointerId: event.pointerId,
        samples: [{ time: event.timeStamp, x: event.clientX }],
        startX: event.clientX,
        startY: event.clientY,
        width: 0,
      };
    },
    [canNavigate, settleTo, stopSettle, writeOffset]
  );

  // A drag of this frame owns its touch outright: a hold armed it (pending), or a grip started it before the finger
  // travelled far enough to swipe.
  const dndListener = useMemo((): DndMonitorListener => {
    const yieldToDrag = (id: UniqueIdentifier): void => {
      const gesture = gestureRef.current;

      if (id !== dragId || !gesture) {
        return;
      }

      gesture.phase = 'declined';
      restoreIfStranded();
    };

    return {
      onDragPending: (event) => yieldToDrag(event.id),
      onDragStart: (event) => yieldToDrag(event.active.id),
    };
  }, [dragId, restoreIfStranded]);

  useDndMonitor(dndListener);

  return {
    contentTrackRef: trackRefCallbacks.content,
    nextTrackRef: trackRefCallbacks.next,
    onPointerDown: handlePointerDown,
    previousTrackRef: trackRefCallbacks.previous,
    /** Render as every track element's `style`; see the landing comment above. */
    restStyle: REST_STYLES[rest.translate],
    showsNeighbors,
    stageRefCallback,
  };
};
