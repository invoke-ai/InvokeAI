import {
  useEffectEvent,
  useImperativeHandle,
  useRef,
  useState,
  type MouseEvent,
  type PointerEvent,
  type Ref,
} from 'react';

/**
 * Lightweight zoom/pan for the preview frame: wheel zooms around the cursor,
 * left-drag pans, double-click toggles fit ⇄ 100%. Implemented as a CSS
 * transform applied imperatively (rAF-batched) to the img inside the fitted
 * frame — high-frequency pointer data never passes through React state; only
 * the rounded zoom percent does, for the corner chip. `scale === 1` is "fit"
 * (the frame's natural CSS size); the chip reports percent of the image's
 * actual pixels. Resets are handled by remounting the frame per image.
 */

/** Max zoom, as a fraction of the image's actual pixel size. */
const MAX_ACTUAL_ZOOM = 8;
/** Wheel-to-zoom sensitivity. */
const WHEEL_ZOOM_FACTOR = 0.0015;
/** Switch to pixelated rendering at/above this actual-pixel zoom. */
const PIXELATED_ACTUAL_ZOOM = 2;

export interface PreviewLoupeControls {
  reset(): void;
  zoomToActual(): void;
}

interface LoupeTransform {
  scale: number;
  tx: number;
  ty: number;
}

const clampAxis = (t: number, frameLen: number, contentLen: number): number =>
  contentLen <= frameLen ? (frameLen - contentLen) / 2 : Math.min(0, Math.max(frameLen - contentLen, t));

export const usePreviewLoupe = ({
  controlsRef,
  enabled,
  naturalWidth,
}: {
  controlsRef?: Ref<PreviewLoupeControls>;
  enabled: boolean;
  naturalWidth: number;
}) => {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const transformRef = useRef<LoupeTransform>({ scale: 1, tx: 0, ty: 0 });
  const rafRef = useRef<number | null>(null);
  const panPointerRef = useRef<{ pointerId: number; startX: number; startY: number; tx: number; ty: number } | null>(
    null
  );
  const [zoomPercent, setZoomPercent] = useState<number | null>(null);

  const getActualZoom = (scale: number): number => {
    const renderedWidth = frameRef.current?.clientWidth ?? 0;

    return renderedWidth > 0 && naturalWidth > 0 ? (scale * renderedWidth) / naturalWidth : scale;
  };

  const apply = useEffectEvent(() => {
    if (rafRef.current !== null) {
      return;
    }

    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const image = imageRef.current;
      const transform = transformRef.current;

      if (!image) {
        return;
      }

      const isFit = transform.scale === 1;

      image.style.transform = isFit ? '' : `translate(${transform.tx}px, ${transform.ty}px) scale(${transform.scale})`;
      image.style.transformOrigin = '0 0';

      const actualZoom = getActualZoom(transform.scale);

      image.style.imageRendering = !isFit && actualZoom >= PIXELATED_ACTUAL_ZOOM ? 'pixelated' : '';
      setZoomPercent(isFit ? null : Math.round(actualZoom * 100));
    });
  });

  const setTransform = useEffectEvent((next: LoupeTransform) => {
    const frame = frameRef.current;

    if (!frame) {
      return;
    }

    const frameWidth = frame.clientWidth;
    const frameHeight = frame.clientHeight;
    const scale = Math.max(1, Math.min(next.scale, (MAX_ACTUAL_ZOOM * naturalWidth) / Math.max(1, frameWidth)));

    transformRef.current =
      scale === 1
        ? { scale: 1, tx: 0, ty: 0 }
        : {
            scale,
            tx: clampAxis(next.tx, frameWidth, frameWidth * scale),
            ty: clampAxis(next.ty, frameHeight, frameHeight * scale),
          };
    apply();
  });

  const zoomAroundPoint = useEffectEvent((pointX: number, pointY: number, nextScale: number) => {
    const { scale, tx, ty } = transformRef.current;
    const ratio = nextScale / scale;

    setTransform({
      scale: nextScale,
      tx: pointX - (pointX - tx) * ratio,
      ty: pointY - (pointY - ty) * ratio,
    });
  });

  const reset = useEffectEvent(() => setTransform({ scale: 1, tx: 0, ty: 0 }));

  const zoomToActual = useEffectEvent(() => {
    const frame = frameRef.current;

    if (!frame || frame.clientWidth === 0) {
      return;
    }

    const scale = Math.max(1, naturalWidth / frame.clientWidth);

    zoomAroundPoint(frame.clientWidth / 2, frame.clientHeight / 2, scale);
  });

  useImperativeHandle(controlsRef, () => ({ reset, zoomToActual }), []);

  // The wheel listener must be attached manually with `passive: false` —
  // React's synthetic wheel events cannot preventDefault. Ref callback with
  // cleanup, so there is no effect to keep in sync.
  const [frameRefCallback] = useState(() => (node: HTMLDivElement | null) => {
    frameRef.current = node;

    if (!node) {
      return;
    }

    const handleWheel = (event: WheelEvent): void => {
      event.preventDefault();
      const rect = node.getBoundingClientRect();
      const { scale } = transformRef.current;
      // Trackpad pinch arrives as ctrl+wheel with finer deltas.
      const sensitivity = event.ctrlKey ? WHEEL_ZOOM_FACTOR * 4 : WHEEL_ZOOM_FACTOR;

      zoomAroundPoint(
        event.clientX - rect.left,
        event.clientY - rect.top,
        scale * Math.exp(-event.deltaY * sensitivity)
      );
    };

    node.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      node.removeEventListener('wheel', handleWheel);

      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  });

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>): void => {
    if (event.button !== 0 || transformRef.current.scale === 1) {
      return;
    }

    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    panPointerRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      tx: transformRef.current.tx,
      ty: transformRef.current.ty,
    };
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>): void => {
    const pan = panPointerRef.current;

    if (!pan || pan.pointerId !== event.pointerId) {
      return;
    }

    event.preventDefault();
    setTransform({
      scale: transformRef.current.scale,
      tx: pan.tx + (event.clientX - pan.startX),
      ty: pan.ty + (event.clientY - pan.startY),
    });
  };

  const handlePointerEnd = (event: PointerEvent<HTMLDivElement>): void => {
    if (panPointerRef.current?.pointerId !== event.pointerId) {
      return;
    }

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    panPointerRef.current = null;
  };

  const handleDoubleClick = (event: MouseEvent<HTMLDivElement>): void => {
    const frame = frameRef.current;

    if (!frame) {
      return;
    }

    if (transformRef.current.scale !== 1) {
      reset();
      return;
    }

    const rect = frame.getBoundingClientRect();
    const scale = Math.max(1, naturalWidth / frame.clientWidth);

    zoomAroundPoint(event.clientX - rect.left, event.clientY - rect.top, scale);
  };

  if (!enabled) {
    return {
      frameProps: null,
      frameRefCallback: null,
      imageRef: null,
      isZoomed: false,
      reset,
      zoomPercent: null,
    };
  }

  return {
    frameProps: {
      onDoubleClick: handleDoubleClick,
      onLostPointerCapture: handlePointerEnd,
      onPointerCancel: handlePointerEnd,
      onPointerDown: handlePointerDown,
      onPointerMove: handlePointerMove,
      onPointerUp: handlePointerEnd,
    },
    frameRefCallback,
    imageRef,
    isZoomed: zoomPercent !== null,
    reset,
    zoomPercent,
  };
};
