import {
  distanceBetween,
  midpointOf,
  panBy,
  pinchZoomAtPoints,
  WHEEL_ZOOM_STEP,
  zoomAtPoint as calculateZoomAtPoint,
  type PanZoomPoint,
} from '@workbench/panZoom';
import {
  useCallback,
  useImperativeHandle,
  useRef,
  useState,
  type MouseEvent,
  type PointerEvent as ReactPointerEvent,
  type Ref,
} from 'react';

import { capturePointer, releasePointer, trackPointerDown } from './loupeGestures';

/**
 * Use the whole stage for wheel zoom, drag pan, fit/100% toggle, and touch pinch. Apply rAF-batched CSS transforms
 * outside React; publish only rounded actual-pixel zoom, with scale 1 meaning fit.
 */

/** Max zoom, as a fraction of the image's actual pixel size. */
const MAX_ACTUAL_ZOOM = 8;
/** Switch to pixelated rendering at/above this actual-pixel zoom. */
const PIXELATED_ACTUAL_ZOOM = 2;

export interface PreviewLoupeControls {
  reset(): void;
  /** Zoom to a fraction of the image's own pixels (1 = 100%) about the stage centre, never below fit. */
  zoomTo(actualZoom: number): void;
  zoomToActual(): void;
}

/** What the loupe shows, for a readout outside the stage: percents of the image's own pixels. */
export interface PreviewZoomState {
  fitPercent: number | null;
  isZoomed: boolean;
  percent: number | null;
}

interface LoupeTransform {
  scale: number;
  /** Stage-space translation applied to the content box (origin 0 0). */
  tx: number;
  ty: number;
}

/** Capture transform, separation, midpoint, and content origin once so pinch moves resolve from a fixed baseline. */
interface PinchGesture {
  /** Client-space position of the content box's untransformed origin. */
  originLeft: number;
  originTop: number;
  pointerIds: [number, number];
  /** Midpoint at gesture start, in content-box space. */
  startCenter: PanZoomPoint;
  startDistance: number;
  startTransform: LoupeTransform;
}

/**
 * Clamp one axis of the translation: content smaller than the stage is
 * centered; content larger may pan but never leaves a gap at either edge.
 * `base` is the content's untransformed layout offset within the stage.
 */
const clampAxis = (t: number, stageLen: number, base: number, scaledLen: number): number =>
  scaledLen <= stageLen ? (stageLen - scaledLen) / 2 - base : Math.min(-base, Math.max(stageLen - scaledLen - base, t));

export const usePreviewLoupe = ({
  controlsRef,
  enabled,
  naturalWidth,
  onZoomChange,
}: {
  controlsRef?: Ref<PreviewLoupeControls>;
  enabled: boolean;
  naturalWidth: number;
  onZoomChange?: (state: PreviewZoomState) => void;
}) => {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const transformRef = useRef<LoupeTransform>({ scale: 1, tx: 0, ty: 0 });
  const rafRef = useRef<number | null>(null);
  const panPointerRef = useRef<{ pointerId: number; startX: number; startY: number; tx: number; ty: number } | null>(
    null
  );
  /** Every pointer currently down on the stage, in client space, in arrival order. */
  const pointersRef = useRef(new Map<number, PanZoomPoint>());
  const pinchRef = useRef<PinchGesture | null>(null);
  /** Whether the touch session's first pointer landed on the draggable image. */
  const dragCandidateRef = useRef(false);
  const lastSourceTokenRef = useRef<string | null | undefined>(undefined);
  const [zoomState, setZoomState] = useState<PreviewZoomState>(FIT_UNMEASURED);
  const publishedRef = useRef<PreviewZoomState>(FIT_UNMEASURED);
  // Keep gesture/ref identity stable across images; callback cleanup must not cancel the frame clearing an old
  // transform.
  const naturalWidthRef = useRef(naturalWidth);
  // eslint-disable-next-line react/refs
  naturalWidthRef.current = naturalWidth;
  const onZoomChangeRef = useRef(onZoomChange);
  // eslint-disable-next-line react/refs
  onZoomChangeRef.current = onZoomChange;

  /** The fitted (untransformed) content's width as a fraction of the image's own pixels. */
  const measureFitRatio = useCallback((): number | null => {
    const renderedWidth = contentRef.current?.clientWidth ?? 0;
    const width = naturalWidthRef.current;

    return renderedWidth > 0 && width > 0 ? renderedWidth / width : null;
  }, []);

  /** One place turns a transform into the readout, so the header and `isZoomed` never disagree. */
  const publish = useCallback(
    (scale: number) => {
      const fitRatio = measureFitRatio();
      const next: PreviewZoomState = {
        fitPercent: fitRatio === null ? null : Math.round(fitRatio * 100),
        isZoomed: scale !== 1,
        percent: fitRatio === null ? null : Math.round(fitRatio * scale * 100),
      };
      const previous = publishedRef.current;

      if (
        previous.fitPercent === next.fitPercent &&
        previous.isZoomed === next.isZoomed &&
        previous.percent === next.percent
      ) {
        return;
      }

      publishedRef.current = next;
      setZoomState(next);
      onZoomChangeRef.current?.(next);
    },
    [measureFitRatio]
  );

  const getActualZoom = useCallback((scale: number): number => {
    const renderedWidth = contentRef.current?.clientWidth ?? 0;
    const width = naturalWidthRef.current;

    return renderedWidth > 0 && width > 0 ? (scale * renderedWidth) / width : scale;
  }, []);

  const apply = useCallback(() => {
    if (rafRef.current !== null) {
      return;
    }

    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const content = contentRef.current;
      const transform = transformRef.current;

      if (!content) {
        return;
      }

      const isFit = transform.scale === 1;

      content.style.transform = isFit
        ? ''
        : `translate(${transform.tx}px, ${transform.ty}px) scale(${transform.scale})`;
      content.style.transformOrigin = '0 0';

      const actualZoom = getActualZoom(transform.scale);

      // `image-rendering` is inherited, so setting it on the content box
      // reaches the img (whose own style leaves it unset while the loupe is
      // enabled).
      content.style.imageRendering = !isFit && actualZoom >= PIXELATED_ACTUAL_ZOOM ? 'pixelated' : '';
      publish(transform.scale);
    });
  }, [getActualZoom, publish]);

  /**
   * Reset transforms when the displayed-image token changes without remounting decoded media; null disables the
   * loupe.
   */
  const syncDisplayedSource = (token: string | null): void => {
    if (lastSourceTokenRef.current === token) {
      return;
    }

    lastSourceTokenRef.current = token;
    // Discard old-image gestures even at fit so remaining touches cannot transform the new image.
    panPointerRef.current = null;
    pinchRef.current = null;
    transformRef.current = { scale: 1, tx: 0, ty: 0 };
    // Schedule reset through apply so intervening wheel updates win and fitted percent uses current layout.
    apply();
  };

  const setTransform = useCallback(
    (next: LoupeTransform) => {
      const stage = stageRef.current;
      const content = contentRef.current;

      if (!stage || !content || content.clientWidth === 0) {
        return;
      }

      const scale = next.scale;

      transformRef.current =
        scale === 1
          ? { scale: 1, tx: 0, ty: 0 }
          : {
              scale,
              tx: clampAxis(next.tx, stage.clientWidth, content.offsetLeft, content.clientWidth * scale),
              ty: clampAxis(next.ty, stage.clientHeight, content.offsetTop, content.clientHeight * scale),
            };
      apply();
    },
    [apply]
  );

  /** Never below fit, never past `MAX_ACTUAL_ZOOM` of the image's own pixels. */
  const constrainScale = useCallback((scale: number): number => {
    const renderedWidth = contentRef.current?.clientWidth ?? 0;
    const maxScale = renderedWidth > 0 ? Math.max(1, (MAX_ACTUAL_ZOOM * naturalWidthRef.current) / renderedWidth) : 1;

    return Math.max(1, Math.min(scale, maxScale));
  }, []);

  /** Zoom keeping the content point under the given stage-space coordinates fixed. */
  const zoomAroundPoint = useCallback(
    (stageX: number, stageY: number, nextScale: number) => {
      const content = contentRef.current;

      if (!content) {
        return;
      }

      const { scale, tx, ty } = transformRef.current;
      const next = calculateZoomAtPoint(
        { pan: { x: tx, y: ty }, zoom: scale },
        nextScale,
        { x: stageX - content.offsetLeft, y: stageY - content.offsetTop },
        constrainScale
      );

      setTransform({
        scale: next.zoom,
        tx: next.pan.x,
        ty: next.pan.y,
      });
    },
    [constrainScale, setTransform]
  );

  // Both commands drop a gesture in flight: a live pinch resolves every move
  // from the transform it started on, so its next move would undo the command.
  const reset = useCallback(() => {
    pinchRef.current = null;
    setTransform({ scale: 1, tx: 0, ty: 0 });
  }, [setTransform]);

  const zoomTo = useCallback(
    (actualZoom: number) => {
      const stage = stageRef.current;
      const content = contentRef.current;

      if (!stage || !content || content.clientWidth === 0) {
        return;
      }

      pinchRef.current = null;
      zoomAroundPoint(
        stage.clientWidth / 2,
        stage.clientHeight / 2,
        Math.max(1, (actualZoom * naturalWidthRef.current) / content.clientWidth)
      );
    },
    [zoomAroundPoint]
  );

  const zoomToActual = useCallback(() => zoomTo(1), [zoomTo]);

  useImperativeHandle(controlsRef, () => ({ reset, zoomTo, zoomToActual }), [reset, zoomTo, zoomToActual]);

  // Reapply transform and remeasure zoom when fitted content resizes or reattaches so DOM and readout agree.
  const contentRefCallback = useCallback(
    (node: HTMLDivElement | null) => {
      contentRef.current = node;

      if (!node || typeof ResizeObserver === 'undefined') {
        return;
      }

      const observer = new ResizeObserver(() => apply());

      observer.observe(node);

      return () => {
        observer.disconnect();
      };
    },
    [apply]
  );

  /** Starts a pan from the given pointer's current position, at the current transform. */
  const beginPan = useCallback((pointerId: number, from: PanZoomPoint): void => {
    panPointerRef.current = {
      pointerId,
      startX: from.x,
      startY: from.y,
      tx: transformRef.current.tx,
      ty: transformRef.current.ty,
    };
  }, []);

  /** Capture both pinch pointers; reject unavailable stage geometry or coincident fingers. */
  const beginPinch = useCallback((pointerIds: [number, number]): PinchGesture | null => {
    const stage = stageRef.current;
    const content = contentRef.current;
    const first = pointersRef.current.get(pointerIds[0]);
    const second = pointersRef.current.get(pointerIds[1]);

    if (!stage || !content || content.clientWidth === 0 || !first || !second) {
      return null;
    }

    const distance = distanceBetween(first, second);

    if (distance === 0) {
      return null;
    }

    const rect = stage.getBoundingClientRect();
    const originLeft = rect.left + content.offsetLeft;
    const originTop = rect.top + content.offsetTop;
    const center = midpointOf(first, second);

    panPointerRef.current = null;
    pinchRef.current = {
      originLeft,
      originTop,
      pointerIds,
      startCenter: { x: center.x - originLeft, y: center.y - originTop },
      startDistance: distance,
      startTransform: transformRef.current,
    };

    for (const pointerId of pointerIds) {
      capturePointer(stage, pointerId);
    }

    return pinchRef.current;
  }, []);

  /**
   * Ends one pointer, wherever it lifted. Lifting one finger of a three-finger
   * gesture re-pinches on what is left; lifting to a single finger hands the
   * gesture over to a pan, so the image keeps following that finger without a
   * release and re-touch.
   */
  const endPointer = useCallback(
    (pointerId: number): void => {
      const pointers = pointersRef.current;
      const stage = stageRef.current;

      if (pointers.delete(pointerId) && stage) {
        releasePointer(stage, pointerId);
      }

      const pinch = pinchRef.current;

      if (pinch?.pointerIds.includes(pointerId)) {
        pinchRef.current = null;
        const remaining = [...pointers.keys()];

        if (remaining.length >= 2) {
          beginPinch([remaining[0]!, remaining[1]!]);
          return;
        }

        const last = remaining[0];
        const lastPoint = last === undefined ? undefined : pointers.get(last);

        if (last !== undefined && lastPoint && transformRef.current.scale !== 1) {
          beginPan(last, lastPoint);
        }

        return;
      }

      if (panPointerRef.current?.pointerId !== pointerId) {
        return;
      }

      panPointerRef.current = null;
    },
    [beginPan, beginPinch]
  );

  // Attach non-passive wheel handling with ref cleanup because React synthetic wheel cannot prevent defaults.
  const stageRefCallback = useCallback(
    (node: HTMLDivElement | null) => {
      stageRef.current = node;

      if (!node) {
        return;
      }

      const handleWheel = (event: WheelEvent): void => {
        event.preventDefault();
        const rect = node.getBoundingClientRect();
        const { scale } = transformRef.current;
        // Trackpad pinch arrives as ctrl+wheel with finer deltas.
        const sensitivity = event.ctrlKey ? WHEEL_ZOOM_STEP * 4 : WHEEL_ZOOM_STEP;

        zoomAroundPoint(
          event.clientX - rect.left,
          event.clientY - rect.top,
          scale * Math.exp(-event.deltaY * sensitivity)
        );
      };

      // Track pointer positions/releases document-wide so off-stage movement cannot leave phantom pinch origins;
      // stage handlers still drive gestures.
      const handleDocumentPointerMove = (event: PointerEvent): void => {
        const pointers = pointersRef.current;

        if (pointers.has(event.pointerId)) {
          pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        }
      };
      const handleDocumentPointerEnd = (event: PointerEvent): void => endPointer(event.pointerId);
      const ownerDocument = node.ownerDocument;

      node.addEventListener('wheel', handleWheel, { passive: false });
      ownerDocument.addEventListener('pointermove', handleDocumentPointerMove, { passive: true });
      ownerDocument.addEventListener('pointerup', handleDocumentPointerEnd);
      ownerDocument.addEventListener('pointercancel', handleDocumentPointerEnd);

      return () => {
        node.removeEventListener('wheel', handleWheel);
        ownerDocument.removeEventListener('pointermove', handleDocumentPointerMove);
        ownerDocument.removeEventListener('pointerup', handleDocumentPointerEnd);
        ownerDocument.removeEventListener('pointercancel', handleDocumentPointerEnd);

        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current);
          rafRef.current = null;
        }
      };
    },
    [endPointer, zoomAroundPoint]
  );

  /**
   * Cancel dnd's pending/active image drag when pinch takes over using a document-targeted non-bubbling
   * pointercancel. Only do so when a finger began on the image, leaving stage handlers untouched.
   */
  const cancelPointerDrag = (): void => {
    const stage = stageRef.current;

    if (stage && dragCandidateRef.current) {
      stage.ownerDocument.dispatchEvent(new PointerEvent('pointercancel'));
    }
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (event.button !== 0) {
      return;
    }

    const pointers = pointersRef.current;

    if (event.isPrimary) {
      pinchRef.current = null;
    }

    const pair = trackPointerDown(pointers, event);

    if (pointers.size === 1) {
      dragCandidateRef.current = contentRef.current?.contains(event.target as Node) ?? false;
    }

    if (pair && !pinchRef.current) {
      event.preventDefault();
      // Two fingers are never a pan, whether or not the pinch can be armed.
      panPointerRef.current = null;

      if (beginPinch(pair)) {
        cancelPointerDrag();
      }

      return;
    }

    if (pointers.size !== 1 || transformRef.current.scale === 1) {
      return;
    }

    event.preventDefault();
    capturePointer(event.currentTarget, event.pointerId);
    beginPan(event.pointerId, { x: event.clientX, y: event.clientY });
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>): void => {
    const pointers = pointersRef.current;

    if (pointers.has(event.pointerId)) {
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    }

    const pinch = pinchRef.current;

    if (pinch) {
      const first = pointers.get(pinch.pointerIds[0]);
      const second = pointers.get(pinch.pointerIds[1]);

      if (!pinch.pointerIds.includes(event.pointerId) || !first || !second) {
        return;
      }

      event.preventDefault();
      const center = midpointOf(first, second);
      const next = pinchZoomAtPoints(
        { pan: { x: pinch.startTransform.tx, y: pinch.startTransform.ty }, zoom: pinch.startTransform.scale },
        {
          center: { x: center.x - pinch.originLeft, y: center.y - pinch.originTop },
          distance: distanceBetween(first, second),
          startCenter: pinch.startCenter,
          startDistance: pinch.startDistance,
        },
        constrainScale
      );

      setTransform({ scale: next.zoom, tx: next.pan.x, ty: next.pan.y });
      return;
    }

    const pan = panPointerRef.current;

    if (!pan || pan.pointerId !== event.pointerId) {
      return;
    }

    event.preventDefault();
    const next = panBy(
      { pan: { x: pan.tx, y: pan.ty }, zoom: transformRef.current.scale },
      { x: event.clientX - pan.startX, y: event.clientY - pan.startY }
    );
    setTransform({ scale: next.zoom, tx: next.pan.x, ty: next.pan.y });
  };

  const handleDoubleClick = (event: MouseEvent<HTMLDivElement>): void => {
    const stage = stageRef.current;
    const content = contentRef.current;

    if (!stage || !content || content.clientWidth === 0) {
      return;
    }

    if (transformRef.current.scale !== 1) {
      reset();
      return;
    }

    const rect = stage.getBoundingClientRect();

    zoomAroundPoint(
      event.clientX - rect.left,
      event.clientY - rect.top,
      Math.max(1, naturalWidth / content.clientWidth)
    );
  };

  if (!enabled) {
    return {
      contentRefCallback: null,
      isZoomed: false,
      reset,
      stageProps: null,
      stageRefCallback: null,
      syncDisplayedSource,
    };
  }

  return {
    contentRefCallback,
    isZoomed: zoomState.isZoomed,
    reset,
    stageProps: {
      onDoubleClick: handleDoubleClick,
      onPointerDown: handlePointerDown,
      onPointerMove: handlePointerMove,
    },
    stageRefCallback,
    syncDisplayedSource,
  };
};

const FIT_UNMEASURED: PreviewZoomState = { fitPercent: null, isZoomed: false, percent: null };
