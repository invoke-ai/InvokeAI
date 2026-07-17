import { panBy, WHEEL_ZOOM_STEP, zoomAtPoint as calculateZoomAtPoint } from '@workbench/panZoom';
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
 * Lightweight zoom/pan for the preview: wheel zooms around the cursor,
 * left-drag pans, double-click toggles fit ⇄ 100%. The *stage* (the dot-grid
 * area) is the viewport — the fitted, framed image scales and pans across the
 * whole stage and clips at its edges, instead of being inspected through its
 * own small wrapper. Implemented as a CSS transform applied imperatively
 * (rAF-batched) to the fitted content box; high-frequency pointer data never
 * passes through React state — only the rounded zoom percent does, for the
 * corner chip. `scale === 1` is "fit"; the chip reports percent of the image's
 * actual pixels.
 */

/** Max zoom, as a fraction of the image's actual pixel size. */
const MAX_ACTUAL_ZOOM = 8;
/** Switch to pixelated rendering at/above this actual-pixel zoom. */
const PIXELATED_ACTUAL_ZOOM = 2;

export interface PreviewLoupeControls {
  reset(): void;
  zoomToActual(): void;
}

interface LoupeTransform {
  scale: number;
  /** Stage-space translation applied to the content box (origin 0 0). */
  tx: number;
  ty: number;
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
}: {
  controlsRef?: Ref<PreviewLoupeControls>;
  enabled: boolean;
  naturalWidth: number;
}) => {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const transformRef = useRef<LoupeTransform>({ scale: 1, tx: 0, ty: 0 });
  const rafRef = useRef<number | null>(null);
  const panPointerRef = useRef<{ pointerId: number; startX: number; startY: number; tx: number; ty: number } | null>(
    null
  );
  const lastSourceTokenRef = useRef<string | null | undefined>(undefined);
  const [zoomPercent, setZoomPercent] = useState<number | null>(null);

  const getActualZoom = (scale: number): number => {
    const renderedWidth = contentRef.current?.clientWidth ?? 0;

    return renderedWidth > 0 && naturalWidth > 0 ? (scale * renderedWidth) / naturalWidth : scale;
  };

  const apply = useEffectEvent(() => {
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
      setZoomPercent(isFit ? null : Math.round(actualZoom * 100));
    });
  });

  /**
   * Called during render with a token identifying the displayed image (or null
   * while the loupe is inapplicable, e.g. live frames). A token change resets
   * the transform in place — no remount, so the img element (and its decoded
   * pixels) survive selection changes without a flash.
   */
  const syncDisplayedSource = (token: string | null): void => {
    if (lastSourceTokenRef.current === token) {
      return;
    }

    lastSourceTokenRef.current = token;

    if (transformRef.current.scale === 1) {
      return;
    }

    transformRef.current = { scale: 1, tx: 0, ty: 0 };
    panPointerRef.current = null;

    if (rafRef.current === null) {
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        const content = contentRef.current;

        if (content) {
          content.style.transform = '';
          content.style.imageRendering = '';
        }

        setZoomPercent(null);
      });
    }
  };

  const setTransform = useEffectEvent((next: LoupeTransform) => {
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
  });

  /** Zoom keeping the content point under the given stage-space coordinates fixed. */
  const zoomAroundPoint = useEffectEvent((stageX: number, stageY: number, nextScale: number) => {
    const content = contentRef.current;

    if (!content) {
      return;
    }

    const { scale, tx, ty } = transformRef.current;
    const maxScale = Math.max(1, (MAX_ACTUAL_ZOOM * naturalWidth) / content.clientWidth);
    const next = calculateZoomAtPoint(
      { pan: { x: tx, y: ty }, zoom: scale },
      nextScale,
      { x: stageX - content.offsetLeft, y: stageY - content.offsetTop },
      (zoom) => Math.max(1, Math.min(zoom, maxScale))
    );

    setTransform({
      scale: next.zoom,
      tx: next.pan.x,
      ty: next.pan.y,
    });
  });

  const reset = useEffectEvent(() => setTransform({ scale: 1, tx: 0, ty: 0 }));

  const zoomToActual = useEffectEvent(() => {
    const stage = stageRef.current;
    const content = contentRef.current;

    if (!stage || !content || content.clientWidth === 0) {
      return;
    }

    zoomAroundPoint(stage.clientWidth / 2, stage.clientHeight / 2, Math.max(1, naturalWidth / content.clientWidth));
  });

  useImperativeHandle(controlsRef, () => ({ reset, zoomToActual }), []);

  // The wheel listener must be attached manually with `passive: false` —
  // React's synthetic wheel events cannot preventDefault. Ref callback with
  // cleanup, so there is no effect to keep in sync.
  const [stageRefCallback] = useState(() => (node: HTMLDivElement | null) => {
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
    const next = panBy(
      { pan: { x: pan.tx, y: pan.ty }, zoom: transformRef.current.scale },
      { x: event.clientX - pan.startX, y: event.clientY - pan.startY }
    );
    setTransform({ scale: next.zoom, tx: next.pan.x, ty: next.pan.y });
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
      contentRef: null,
      isZoomed: false,
      reset,
      stageProps: null,
      stageRefCallback: null,
      syncDisplayedSource,
      zoomPercent: null,
    };
  }

  return {
    contentRef,
    isZoomed: zoomPercent !== null,
    reset,
    stageProps: {
      onDoubleClick: handleDoubleClick,
      onLostPointerCapture: handlePointerEnd,
      onPointerCancel: handlePointerEnd,
      onPointerDown: handlePointerDown,
      onPointerMove: handlePointerMove,
      onPointerUp: handlePointerEnd,
    },
    stageRefCallback,
    syncDisplayedSource,
    zoomPercent,
  };
};
