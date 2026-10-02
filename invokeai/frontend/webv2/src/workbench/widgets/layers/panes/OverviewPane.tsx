import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';

import { Box, chakra, Flex } from '@chakra-ui/react';
import { useCanvasEngine, type CanvasEngineHandle } from '@workbench/canvas-operations/react';
import { useCanvasLayerPixelEpoch } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const KEY_PAN_STEP_PX = 40;
/** Floor between composite repaints; content changes inside the window coalesce into one trailing draw. */
const REDRAW_INTERVAL_MS = 100;
/** The outline updates on this cadence and transitions between positions, so tracking reads as a glide. */
const VIEW_RECT_INTERVAL_MS = 80;

const FOCUS_PROPS = { outline: '2px solid {colors.accent.solid}', outlineOffset: '2px' };

const CHECKERBOARD_CSS = {
  backgroundImage: 'repeating-conic-gradient({colors.bg.emphasized} 0% 25%, {colors.bg.inset} 0% 50%)',
  backgroundSize: '12px 12px',
};

const VIEW_RECT_TRANSITION = `left ${VIEW_RECT_INTERVAL_MS}ms linear, top ${VIEW_RECT_INTERVAL_MS}ms linear, width ${VIEW_RECT_INTERVAL_MS}ms linear, height ${VIEW_RECT_INTERVAL_MS}ms linear`;

/**
 * Show a fitted composite with viewport outline; click/drag centers, wheel zooms, and arrows pan. Throttle
 * painting and update the outline imperatively.
 */
export const OverviewPane = () => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();

  if (!engine) {
    return (
      <Flex align="center" color="fg.muted" fontSize="xs" h="full" justify="center" p="4">
        {t('widgets.properties.noCanvas')}
      </Flex>
    );
  }
  return <ConnectedOverview engine={engine} />;
};

interface OverviewFrame {
  /** Preview pixels per document unit. */
  scale: number;
  widthPx: number;
  heightPx: number;
}

const ConnectedOverview = ({ engine }: { engine: CanvasEngineHandle }) => {
  const { t } = useTranslation();
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const viewRectRef = useRef<HTMLDivElement>(null);
  const drag = useRef<AbortController | null>(null);
  const [hostSize, setHostSize] = useState<{ width: number; height: number }>({ height: 0, width: 0 });

  const pixelEpoch = useCanvasLayerPixelEpoch(engine);
  const stacks = useActiveProjectSelector((project) => project.canvas.document.stacks);
  const docWidth = useActiveProjectSelector((project) => project.canvas.document.width);
  const docHeight = useActiveProjectSelector((project) => project.canvas.document.height);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) {
      return;
    }
    const observer = new ResizeObserver((observations) => {
      const content = observations[0]?.contentRect;
      if (content) {
        setHostSize({ height: content.height, width: content.width });
      }
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  useEffect(() => () => drag.current?.abort(), []);

  // Mirrors the engine's `fitThumbnailSize` math, so the drawn bitmap and the
  // outline overlay agree without a state round trip.
  const frame = useMemo<OverviewFrame | null>(() => {
    const maxSize = Math.floor(Math.min(hostSize.width, hostSize.height));
    if (maxSize < 16 || docWidth <= 0 || docHeight <= 0) {
      return null;
    }
    const scale = Math.min(1, maxSize / docWidth, maxSize / docHeight);
    return {
      heightPx: Math.max(1, Math.round(docHeight * scale)),
      scale,
      widthPx: Math.max(1, Math.round(docWidth * scale)),
    };
  }, [docHeight, docWidth, hostSize]);
  const frameRef = useRef(frame);
  useLayoutEffect(() => {
    frameRef.current = frame;
  }, [frame]);

  const redraw = useRef({ last: 0, timer: null as ReturnType<typeof setTimeout> | null });
  // Only a real composite consumes throttle budget, so the pre-measure clearing
  // draw never defers the first paint.
  const drawComposite = useCallback((): boolean => {
    const canvas = canvasRef.current;
    const current = frameRef.current;
    if (!canvas) {
      return false;
    }
    if (!current) {
      canvas.width = 0;
      canvas.height = 0;
      return false;
    }
    engine.previews.drawDocumentOverview(canvas, Math.max(current.widthPx, current.heightPx));
    return true;
  }, [engine]);
  const scheduleRedraw = useCallback(() => {
    const state = redraw.current;
    if (state.timer !== null) {
      return;
    }
    const elapsed = Date.now() - state.last;
    if (elapsed >= REDRAW_INTERVAL_MS) {
      if (drawComposite()) {
        state.last = Date.now();
      }
      return;
    }
    state.timer = setTimeout(() => {
      state.timer = null;
      if (drawComposite()) {
        state.last = Date.now();
      }
    }, REDRAW_INTERVAL_MS - elapsed);
  }, [drawComposite]);
  useEffect(() => {
    const state = redraw.current;
    return () => {
      if (state.timer !== null) {
        clearTimeout(state.timer);
      }
    };
  }, []);
  useEffect(() => {
    scheduleRedraw();
  }, [pixelEpoch, stacks, frame, scheduleRedraw]);

  // Throttle DOM outline updates and let CSS interpolate between them.
  useEffect(() => {
    const viewport = engine.viewport.getViewport();
    let timer: ReturnType<typeof setTimeout> | null = null;
    let last = 0;
    const apply = () => {
      const element = viewRectRef.current;
      const current = frameRef.current;
      if (!element) {
        return;
      }
      const size = viewport.getViewportSize();
      if (!current || size.width <= 0 || size.height <= 0) {
        element.style.visibility = 'hidden';
        return;
      }
      const topLeft = viewport.screenToDocument({ x: 0, y: 0 });
      const bottomRight = viewport.screenToDocument({ x: size.width, y: size.height });
      const width = (bottomRight.x - topLeft.x) * current.scale;
      const height = (bottomRight.y - topLeft.y) * current.scale;
      element.style.visibility = 'visible';
      element.style.left = `${Math.max(-current.widthPx, Math.min(topLeft.x * current.scale, current.widthPx))}px`;
      element.style.top = `${Math.max(-current.heightPx, Math.min(topLeft.y * current.scale, current.heightPx))}px`;
      element.style.width = `${Math.max(4, Math.min(width, current.widthPx * 2))}px`;
      element.style.height = `${Math.max(4, Math.min(height, current.heightPx * 2))}px`;
    };
    const update = () => {
      if (timer !== null) {
        return;
      }
      const elapsed = Date.now() - last;
      if (elapsed >= VIEW_RECT_INTERVAL_MS) {
        last = Date.now();
        apply();
        return;
      }
      timer = setTimeout(() => {
        timer = null;
        last = Date.now();
        apply();
      }, VIEW_RECT_INTERVAL_MS - elapsed);
    };
    apply();
    const unsubscribe = viewport.subscribe(update);
    return () => {
      unsubscribe();
      if (timer !== null) {
        clearTimeout(timer);
      }
    };
  }, [engine, frame]);

  const centerOn = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      const current = frameRef.current;
      if (!canvas || !current) {
        return;
      }
      const bounds = canvas.getBoundingClientRect();
      const documentPoint = {
        x: (clientX - bounds.left) / current.scale,
        y: (clientY - bounds.top) / current.scale,
      };
      const viewport = engine.viewport.getViewport();
      const size = viewport.getViewportSize();
      const screenPoint = viewport.documentToScreen(documentPoint);
      viewport.panBy({ x: size.width / 2 - screenPoint.x, y: size.height / 2 - screenPoint.y });
    },
    [engine]
  );

  // React registers wheel listeners passively; zooming must preventDefault, so
  // the listener is attached natively.
  useEffect(() => {
    const button = buttonRef.current;
    if (!button) {
      return;
    }
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const canvas = canvasRef.current;
      const current = frameRef.current;
      if (!canvas || !current) {
        return;
      }
      const bounds = canvas.getBoundingClientRect();
      const documentPoint = {
        x: (event.clientX - bounds.left) / current.scale,
        y: (event.clientY - bounds.top) / current.scale,
      };
      const viewport = engine.viewport.getViewport();
      viewport.wheelZoom(event.deltaY, viewport.documentToScreen(documentPoint));
    };
    button.addEventListener('wheel', handleWheel, { passive: false });
    return () => button.removeEventListener('wheel', handleWheel);
  }, [engine]);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      event.currentTarget.focus();
      const controller = new AbortController();
      drag.current?.abort();
      drag.current = controller;
      centerOn(event.clientX, event.clientY);
      window.addEventListener('pointermove', (moveEvent) => centerOn(moveEvent.clientX, moveEvent.clientY), {
        signal: controller.signal,
      });
      window.addEventListener('pointerup', () => controller.abort(), { signal: controller.signal });
      window.addEventListener('pointercancel', () => controller.abort(), { signal: controller.signal });
    },
    [centerOn]
  );
  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLButtonElement>) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        engine.viewport.fitToView();
        return;
      }
      const step = event.shiftKey ? KEY_PAN_STEP_PX * 4 : KEY_PAN_STEP_PX;
      const delta =
        event.key === 'ArrowLeft'
          ? { x: step, y: 0 }
          : event.key === 'ArrowRight'
            ? { x: -step, y: 0 }
            : event.key === 'ArrowUp'
              ? { x: 0, y: step }
              : event.key === 'ArrowDown'
                ? { x: 0, y: -step }
                : null;
      if (!delta) {
        return;
      }
      event.preventDefault();
      engine.viewport.getViewport().panBy(delta);
    },
    [engine]
  );
  const fit = useCallback(() => engine.viewport.fitToView(), [engine]);

  return (
    <Flex ref={hostRef} align="center" h="full" justify="center" minH="0" overflow="hidden" p="2">
      <chakra.button
        ref={buttonRef}
        aria-label={t('widgets.layers.overviewPane.pan')}
        css={CHECKERBOARD_CSS}
        cursor="crosshair"
        display="block"
        position="relative"
        rounded="xs"
        type="button"
        _focusVisible={FOCUS_PROPS}
        onDoubleClick={fit}
        onKeyDown={onKeyDown}
        onPointerDown={onPointerDown}
      >
        <chakra.canvas ref={canvasRef} display="block" touchAction="none" />
        <Box
          ref={viewRectRef}
          borderColor="accent.solid"
          borderWidth="1.5px"
          pointerEvents="none"
          position="absolute"
          rounded="xs"
          transition={VIEW_RECT_TRANSITION}
          visibility="hidden"
        />
      </chakra.button>
    </Flex>
  );
};
