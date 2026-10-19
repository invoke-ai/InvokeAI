/* oxlint-disable react-perf/jsx-no-new-function-as-prop -- the container ref callback is intentionally re-created when `engine` changes, so a project switch detaches the old engine and attaches the new one. */
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react';

import { Box } from '@chakra-ui/react';
import { isResizeDragActive, subscribeResizeDrag } from '@platform/ui/ResizeHandle';
import { shouldFocusCanvasSurface } from '@workbench/widgets/canvas/surfaceFocus';
import { TextEditPortal } from '@workbench/widgets/canvas/TextEditPortal';
import { useRef } from 'react';

export type CanvasSurfaceEngine = Pick<
  CanvasEngineHandle,
  'document' | 'interaction' | 'layers' | 'surface' | 'viewport'
>;

/**
 * Focus the canvas container in capture before engine handlers so hotkeys resolve here. Preserve focus already
 * inside, especially text editing, to avoid blur-commit before the engine's commit-and-swallow path.
 */
const focusCanvasSurface = (event: ReactPointerEvent<HTMLDivElement>) => {
  if (shouldFocusCanvasSurface(event.currentTarget, event.target, document.activeElement)) {
    event.currentTarget.focus({ preventScroll: true });
  }
};

/**
 * Bind document/overlay canvases and ResizeObserver through an engine-keyed ref callback with cleanup. The engine
 * owns input without React interaction renders.
 */
export const CanvasSurface = ({ engine }: { engine: CanvasSurfaceEngine }) => {
  const screenRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const bindContainer = (container: HTMLDivElement) => {
    const screen = screenRef.current;
    const overlay = overlayRef.current;
    if (!screen || !overlay) {
      return;
    }

    engine.surface.attach(screen, overlay, container);

    // A handle drag defers the resize and full recomposition to its release; pixel CSS sizes make the interim
    // canvas crop or reveal rather than stretch.
    let isSizeStale = false;
    let syncedSize = '';
    const syncSize = (canDefer: boolean) => {
      if (canDefer && isResizeDragActive()) {
        isSizeStale = true;
        return;
      }
      isSizeStale = false;
      const width = container.clientWidth;
      const height = container.clientHeight;
      const dpr = globalThis.devicePixelRatio || 1;
      // Resizing reallocates and recomposes everything, so an unchanged size is skipped.
      if (canDefer && syncedSize === `${width}x${height}@${dpr}`) {
        return;
      }
      syncedSize = `${width}x${height}@${dpr}`;
      for (const canvas of [screen, overlay]) {
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }
      engine.surface.resize(width, height, dpr);
    };

    // The first sync never waits: the fit below needs a sized viewport.
    syncSize(false);
    // Fit the document into view the first time this canvas is shown, once the
    // viewport is sized. The shell keeps widgets mounted across layout switches,
    // so this callback re-runs on every re-show and an unconditional fit would
    // reset the user's zoom and pan each time they came back.
    engine.viewport.fitToViewOnFirstShow();

    const observer = new ResizeObserver(() => syncSize(true));
    observer.observe(container);
    const unsubscribeResizeDrag = subscribeResizeDrag(() => {
      if (isSizeStale) {
        syncSize(true);
      }
    });

    return () => {
      unsubscribeResizeDrag();
      observer.disconnect();
      engine.surface.detach();
    };
  };

  return (
    <Box
      ref={bindContainer}
      h="full"
      outline="none"
      overflow="hidden"
      position="relative"
      tabIndex={-1}
      w="full"
      onPointerDownCapture={focusCanvasSurface}
    >
      <canvas ref={screenRef} style={CANVAS_STYLE} />
      <canvas ref={overlayRef} style={OVERLAY_STYLE} />
      {/* Position text editing inside the canvas container so documentToScreen offsets share the canvas origin. */}
      <TextEditPortal engine={engine} />
    </Box>
  );
};

const CANVAS_STYLE: CSSProperties = {
  left: 0,
  position: 'absolute',
  top: 0,
  touchAction: 'none',
};

const OVERLAY_STYLE: CSSProperties = { ...CANVAS_STYLE, zIndex: 1 };
