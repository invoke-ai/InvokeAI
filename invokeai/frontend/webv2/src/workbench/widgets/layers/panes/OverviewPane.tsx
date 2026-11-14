import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';

import { Box, chakra, Flex } from '@chakra-ui/react';
import { useCanvasEngine, type CanvasEngineHandle } from '@workbench/canvas-operations/react';
import { useCallback, useRef } from 'react';
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

/**
 * The viewport outline is three 1px bands inside the box, dark/light/dark, matching the canvas brush ring so it reads
 * over any pixels. Fixed black/white rather than theme tokens: contrast must not depend on the content or the theme.
 * Borders, unlike box-shadow spreads, snap to whole device pixels, so each band stays crisp at any device pixel ratio.
 */
const VIEW_RECT_DARK = '#000000';
const VIEW_RECT_LIGHT = '#ffffff';
/** Smallest drawn outline: the three bands on each side plus a 2px hollow, so a tiny viewport still reads as a box. */
const VIEW_RECT_MIN_PX = 8;
const VIEW_RECT_INNER_CSS = {
  _before: {
    borderColor: VIEW_RECT_DARK,
    borderWidth: '1px',
    content: '""',
    inset: '0',
    position: 'absolute',
  },
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
      <Flex align="center" color="fg.muted" fontSize="md" h="full" justify="center" p="4">
        {t('widgets.properties.noCanvas')}
      </Flex>
    );
  }
  return <ConnectedOverview engine={engine} />;
};

/** How the drawn bitmap maps to document space: target pixel = (document point − origin) × scale. */
interface OverviewMapping {
  /** Document-space point drawn at the target's top-left pixel. */
  x: number;
  y: number;
  /** Target pixels per document unit. */
  scale: number;
}

const clientToDocument = (
  mapping: OverviewMapping,
  canvasBounds: DOMRect,
  clientX: number,
  clientY: number
): { x: number; y: number } => ({
  x: mapping.x + (clientX - canvasBounds.left) / mapping.scale,
  y: mapping.y + (clientY - canvasBounds.top) / mapping.scale,
});

const ConnectedOverview = ({ engine }: { engine: CanvasEngineHandle }) => {
  const { t } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const viewRectRef = useRef<HTMLDivElement>(null);
  const drag = useRef<AbortController | null>(null);
  // Written after each draw from the rect the engine reports, so pointer mapping and the outline match the bitmap.
  const mapping = useRef<OverviewMapping | null>(null);

  // Engine-keyed host binding: the host's size, the engine's document and pixel signals, and the viewport drive the
  // bitmap and outline imperatively, without re-rendering the pane.
  const bindHost = useCallback(
    (host: HTMLDivElement | null) => {
      const canvas = canvasRef.current;
      const button = buttonRef.current;
      const outline = viewRectRef.current;
      if (!host || !canvas || !button || !outline) {
        return;
      }
      const viewport = engine.viewport.getViewport();
      let maxSizePx = 0;

      const applyOutline = () => {
        const current = mapping.current;
        const size = viewport.getViewportSize();
        if (!current || size.width <= 0 || size.height <= 0) {
          outline.style.visibility = 'hidden';
          return;
        }
        const widthPx = canvas.width;
        const heightPx = canvas.height;
        const topLeft = viewport.screenToDocument({ x: 0, y: 0 });
        const bottomRight = viewport.screenToDocument({ x: size.width, y: size.height });
        const left = (topLeft.x - current.x) * current.scale;
        const top = (topLeft.y - current.y) * current.scale;
        const width = (bottomRight.x - topLeft.x) * current.scale;
        const height = (bottomRight.y - topLeft.y) * current.scale;
        const drawnWidth = Math.max(VIEW_RECT_MIN_PX, Math.min(width, widthPx * 2));
        const drawnHeight = Math.max(VIEW_RECT_MIN_PX, Math.min(height, heightPx * 2));
        // A box grown to the minimum stays centred on the viewport.
        const drawnLeft = left - Math.max(0, drawnWidth - width) / 2;
        const drawnTop = top - Math.max(0, drawnHeight - height) / 2;
        outline.style.visibility = 'visible';
        outline.style.left = `${Math.max(-widthPx, Math.min(drawnLeft, widthPx))}px`;
        outline.style.top = `${Math.max(-heightPx, Math.min(drawnTop, heightPx))}px`;
        outline.style.width = `${drawnWidth}px`;
        outline.style.height = `${drawnHeight}px`;
      };
      // Throttle DOM outline updates and let CSS interpolate between them.
      const outlineThrottle = { last: 0, timer: null as ReturnType<typeof setTimeout> | null };
      const updateOutline = () => {
        if (outlineThrottle.timer !== null) {
          return;
        }
        const elapsed = Date.now() - outlineThrottle.last;
        if (elapsed >= VIEW_RECT_INTERVAL_MS) {
          outlineThrottle.last = Date.now();
          applyOutline();
          return;
        }
        outlineThrottle.timer = setTimeout(() => {
          outlineThrottle.timer = null;
          outlineThrottle.last = Date.now();
          applyOutline();
        }, VIEW_RECT_INTERVAL_MS - elapsed);
      };

      // Only a real composite consumes throttle budget, so the pre-measure clearing draw never defers the first
      // paint. The outline follows every draw at once, so it never disagrees with the bitmap.
      const drawComposite = (): boolean => {
        const rect = maxSizePx < 16 ? null : engine.previews.drawDocumentOverview(canvas, maxSizePx);
        if (!rect) {
          canvas.width = 0;
          canvas.height = 0;
        }
        mapping.current = rect && canvas.width > 0 ? { scale: canvas.width / rect.width, x: rect.x, y: rect.y } : null;
        applyOutline();
        return rect !== null;
      };
      const redraw = { last: 0, timer: null as ReturnType<typeof setTimeout> | null };
      const scheduleRedraw = () => {
        if (redraw.timer !== null) {
          return;
        }
        const elapsed = Date.now() - redraw.last;
        if (elapsed >= REDRAW_INTERVAL_MS) {
          if (drawComposite()) {
            redraw.last = Date.now();
          }
          return;
        }
        redraw.timer = setTimeout(() => {
          redraw.timer = null;
          if (drawComposite()) {
            redraw.last = Date.now();
          }
        }, REDRAW_INTERVAL_MS - elapsed);
      };

      const observer = new ResizeObserver((observations) => {
        const content = observations[0]?.contentRect;
        if (content) {
          maxSizePx = Math.floor(Math.min(content.width, content.height));
          scheduleRedraw();
        }
      });
      observer.observe(host);

      // Layer and frame edits change the drawn extent; selection-only document changes do not repaint.
      let drawnDocument = engine.document.getDocument();
      const onDocumentChange = () => {
        const doc = engine.document.getDocument();
        if (doc?.stacks !== drawnDocument?.stacks || doc?.bbox !== drawnDocument?.bbox) {
          drawnDocument = doc;
          scheduleRedraw();
        }
      };
      const unsubscribeDocument = engine.interaction.subscribe('documentEpoch', onDocumentChange);
      const unsubscribePixels = engine.interaction.subscribe('layerPixelEpoch', scheduleRedraw);
      const unsubscribeViewport = viewport.subscribe(updateOutline);

      // React registers wheel listeners passively; zooming must preventDefault, so the listener is attached natively.
      const handleWheel = (event: WheelEvent) => {
        event.preventDefault();
        const current = mapping.current;
        if (!current) {
          return;
        }
        const documentPoint = clientToDocument(current, canvas.getBoundingClientRect(), event.clientX, event.clientY);
        viewport.wheelZoom(event.deltaY, viewport.documentToScreen(documentPoint));
      };
      button.addEventListener('wheel', handleWheel, { passive: false });

      drawComposite();
      return () => {
        observer.disconnect();
        unsubscribeDocument();
        unsubscribePixels();
        unsubscribeViewport();
        button.removeEventListener('wheel', handleWheel);
        for (const timer of [redraw.timer, outlineThrottle.timer]) {
          if (timer !== null) {
            clearTimeout(timer);
          }
        }
        mapping.current = null;
        // A drag maps through this engine's viewport, so it ends with the binding.
        drag.current?.abort();
      };
    },
    [engine]
  );

  const centerOn = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      const current = mapping.current;
      if (!canvas || !current) {
        return;
      }
      const documentPoint = clientToDocument(current, canvas.getBoundingClientRect(), clientX, clientY);
      const viewport = engine.viewport.getViewport();
      const size = viewport.getViewportSize();
      const screenPoint = viewport.documentToScreen(documentPoint);
      viewport.panBy({ x: size.width / 2 - screenPoint.x, y: size.height / 2 - screenPoint.y });
    },
    [engine]
  );

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
    <Flex ref={bindHost} align="center" h="full" justify="center" minH="0" overflow="hidden" p="2">
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
          borderColor={VIEW_RECT_DARK}
          borderWidth="1px"
          pointerEvents="none"
          position="absolute"
          transition={VIEW_RECT_TRANSITION}
          visibility="hidden"
        >
          <Box
            borderColor={VIEW_RECT_LIGHT}
            borderWidth="1px"
            css={VIEW_RECT_INNER_CSS}
            inset="0"
            position="absolute"
          />
        </Box>
      </chakra.button>
    </Flex>
  );
};
