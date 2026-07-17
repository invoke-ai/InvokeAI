import { panBy, WHEEL_ZOOM_STEP, zoomAtPoint as calculateZoomAtPoint } from '@workbench/panZoom';
/* eslint-disable react/react-compiler */
import { useEffectEvent, useRef, useState, type MouseEvent, type PointerEvent } from 'react';

/**
 * Shared zoom/pan for the side-by-side comparison panes: one transform, kept
 * in image-fraction space (scale is unitless; translation is a fraction of the
 * pane's rendered size), applied imperatively to every registered pane. Zoom
 * into the left eye and the right pane's left eye follows. Gated by the caller
 * to matching-dimension pairs, where fraction space is exact.
 */

const MAX_ACTUAL_ZOOM = 8;
const PIXELATED_ACTUAL_ZOOM = 2;

interface FractionTransform {
  scale: number;
  fx: number;
  fy: number;
}

interface PaneElements {
  frame: HTMLDivElement | null;
  image: HTMLImageElement | null;
}

export interface CompareLoupePane {
  frameProps: {
    onDoubleClick: (event: MouseEvent<HTMLDivElement>) => void;
    onLostPointerCapture: (event: PointerEvent<HTMLDivElement>) => void;
    onPointerCancel: (event: PointerEvent<HTMLDivElement>) => void;
    onPointerDown: (event: PointerEvent<HTMLDivElement>) => void;
    onPointerMove: (event: PointerEvent<HTMLDivElement>) => void;
    onPointerUp: (event: PointerEvent<HTMLDivElement>) => void;
  };
  frameRefCallback: (node: HTMLDivElement | null) => void;
  imageRefCallback: (node: HTMLImageElement | null) => void;
}

const clampFraction = (f: number, scale: number): number =>
  scale <= 1 ? (1 - scale) / 2 : Math.min(0, Math.max(1 - scale, f));

export const useCompareLoupe = ({
  enabled,
  naturalWidth,
}: {
  enabled: boolean;
  naturalWidth: number;
}): { getPane: (index: 0 | 1) => CompareLoupePane | null; isZoomed: boolean } => {
  const transformRef = useRef<FractionTransform>({ fx: 0, fy: 0, scale: 1 });
  const panesRef = useRef<[PaneElements, PaneElements]>([
    { frame: null, image: null },
    { frame: null, image: null },
  ]);
  const rafRef = useRef<number | null>(null);
  const panRef = useRef<{ pointerId: number; startX: number; startY: number; fx: number; fy: number } | null>(null);
  const [isZoomed, setIsZoomed] = useState(false);

  const apply = useEffectEvent(() => {
    if (rafRef.current !== null) {
      return;
    }

    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const { fx, fy, scale } = transformRef.current;
      const isFit = scale === 1;

      for (const pane of panesRef.current) {
        const { frame, image } = pane;

        if (!frame || !image) {
          continue;
        }

        image.style.transform = isFit
          ? ''
          : `translate(${fx * frame.clientWidth}px, ${fy * frame.clientHeight}px) scale(${scale})`;
        image.style.transformOrigin = '0 0';

        const actualZoom = (scale * frame.clientWidth) / Math.max(1, naturalWidth);

        image.style.imageRendering = !isFit && actualZoom >= PIXELATED_ACTUAL_ZOOM ? 'pixelated' : '';
      }

      setIsZoomed(!isFit);
    });
  });

  const setTransform = useEffectEvent((next: FractionTransform) => {
    const scale = next.scale;

    transformRef.current =
      scale === 1
        ? { fx: 0, fy: 0, scale: 1 }
        : { fx: clampFraction(next.fx, scale), fy: clampFraction(next.fy, scale), scale };
    apply();
  });

  const zoomAroundFraction = useEffectEvent((pfx: number, pfy: number, nextScale: number) => {
    const { fx, fy, scale } = transformRef.current;
    const firstFrame = panesRef.current.find((pane) => pane.frame)?.frame ?? null;
    const maxScale = firstFrame
      ? Math.max(1, (MAX_ACTUAL_ZOOM * naturalWidth) / Math.max(1, firstFrame.clientWidth))
      : MAX_ACTUAL_ZOOM;
    const next = calculateZoomAtPoint({ pan: { x: fx, y: fy }, zoom: scale }, nextScale, { x: pfx, y: pfy }, (zoom) =>
      Math.max(1, Math.min(zoom, maxScale))
    );

    setTransform({
      fx: next.pan.x,
      fy: next.pan.y,
      scale: next.zoom,
    });
  });

  const handleWheel = useEffectEvent((node: HTMLDivElement, event: WheelEvent) => {
    event.preventDefault();
    const rect = node.getBoundingClientRect();

    if (rect.width === 0 || rect.height === 0) {
      return;
    }

    const sensitivity = event.ctrlKey ? WHEEL_ZOOM_STEP * 4 : WHEEL_ZOOM_STEP;

    zoomAroundFraction(
      (event.clientX - rect.left) / rect.width,
      (event.clientY - rect.top) / rect.height,
      transformRef.current.scale * Math.exp(-event.deltaY * sensitivity)
    );
  });

  // One stable set of callbacks per pane index; wheel listeners need manual
  // attachment (`passive: false`) so ref callbacks with cleanup own them.
  const [panes] = useState<[CompareLoupePane, CompareLoupePane]>(() => {
    const createPane = (index: 0 | 1): CompareLoupePane => ({
      frameProps: {
        onDoubleClick: (event) => {
          const frame = panesRef.current[index].frame;

          if (!frame || frame.clientWidth === 0) {
            return;
          }

          if (transformRef.current.scale !== 1) {
            setTransform({ fx: 0, fy: 0, scale: 1 });
            return;
          }

          const rect = frame.getBoundingClientRect();

          zoomAroundFraction(
            (event.clientX - rect.left) / rect.width,
            (event.clientY - rect.top) / rect.height,
            Math.max(1, naturalWidth / frame.clientWidth)
          );
        },
        onLostPointerCapture: (event) => endPan(event),
        onPointerCancel: (event) => endPan(event),
        onPointerDown: (event) => {
          if (event.button !== 0 || transformRef.current.scale === 1) {
            return;
          }

          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          panRef.current = {
            fx: transformRef.current.fx,
            fy: transformRef.current.fy,
            pointerId: event.pointerId,
            startX: event.clientX,
            startY: event.clientY,
          };
        },
        onPointerMove: (event) => {
          const pan = panRef.current;
          const frame = panesRef.current[index].frame;

          if (!pan || pan.pointerId !== event.pointerId || !frame || frame.clientWidth === 0) {
            return;
          }

          event.preventDefault();
          const next = panBy(
            { pan: { x: pan.fx, y: pan.fy }, zoom: transformRef.current.scale },
            {
              x: (event.clientX - pan.startX) / frame.clientWidth,
              y: (event.clientY - pan.startY) / frame.clientHeight,
            }
          );
          setTransform({ fx: next.pan.x, fy: next.pan.y, scale: next.zoom });
        },
        onPointerUp: (event) => endPan(event),
      },
      frameRefCallback: (node) => {
        panesRef.current[index].frame = node;

        if (!node) {
          return;
        }

        const onWheel = (event: WheelEvent): void => handleWheel(node, event);

        node.addEventListener('wheel', onWheel, { passive: false });
        apply();

        return () => {
          node.removeEventListener('wheel', onWheel);
        };
      },
      imageRefCallback: (node) => {
        panesRef.current[index].image = node;

        if (node) {
          apply();
        }
      },
    });
    const endPan = (event: PointerEvent<HTMLDivElement>): void => {
      if (panRef.current?.pointerId !== event.pointerId) {
        return;
      }

      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }

      panRef.current = null;
    };

    return [createPane(0), createPane(1)];
  });

  if (!enabled) {
    return { getPane: () => null, isZoomed: false };
  }

  return { getPane: (index) => panes[index], isZoomed };
};
