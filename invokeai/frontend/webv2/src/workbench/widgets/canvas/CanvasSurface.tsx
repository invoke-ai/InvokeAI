/* oxlint-disable react-perf/jsx-no-new-function-as-prop -- the container ref callback is intentionally re-created when `engine` changes, so a project switch detaches the old engine and attaches the new one. */
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react';

import { Box } from '@chakra-ui/react';
import { shouldFocusCanvasSurface } from '@workbench/widgets/canvas/surfaceFocus';
import { TextEditPortal } from '@workbench/widgets/canvas/TextEditPortal';
import { useRef } from 'react';

/**
 * Give the canvas widget hotkey focus on a pointerdown on the surface, so tool
 * hotkeys (`b`/`e`/`r`/…) work immediately after clicking the canvas.
 *
 * The two `<canvas>` targets aren't focusable, so a plain click never moves DOM
 * focus — it stays on whatever was last focused (e.g. a layers-panel button in a
 * *different* widget), and the hotkey runtime's `getHotkeyTargetWidget` keeps
 * resolving to that widget, so canvas hotkeys don't fire. Focusing this
 * `tabIndex={-1}` container (a descendant of the canvas widget's
 * `data-hotkey-widget-instance-id` element) moves focus into the canvas subtree
 * so hotkeys resolve to the canvas. Focus-from-pointer doesn't match
 * `:focus-visible`, so there's no focus ring.
 *
 * When focus is ALREADY inside the surface — critically, the text tool's
 * contenteditable — this must NOT refocus: doing so in the capture phase would
 * blur-commit the open text session before the engine's own (bubble-phase)
 * pointerdown could run its commit-and-swallow, regressing "click away to
 * commit" into "commit + open a stray new session at the click point". The full
 * decision lives in the node-tested {@link shouldFocusCanvasSurface}.
 *
 * Runs in the capture phase so it fires before the engine's overlay pointerdown
 * listener (which may `stopPropagation`).
 */
const focusCanvasSurface = (event: ReactPointerEvent<HTMLDivElement>) => {
  if (shouldFocusCanvasSurface(event.currentTarget, event.target, document.activeElement)) {
    event.currentTarget.focus({ preventScroll: true });
  }
};

/**
 * The engine-rendered canvas surface: two stacked `<canvas>` targets (the
 * composited document below, the interaction overlay on top) bound to the
 * shared {@link CanvasEngine}.
 *
 * Binding is done through the container's **ref callback** (React 19: the
 * returned function is the cleanup), never a `useEffect` — the callback wires
 * `attach` + a `ResizeObserver` and tears both down on unmount or engine swap.
 * Because the callback closes over `engine`, a project switch (new engine
 * instance) re-runs it: detach the old, attach the new. Pointer/wheel/key input
 * is owned entirely by the engine via the overlay, so this component never
 * re-renders on interaction.
 */
export const CanvasSurface = ({ engine }: { engine: CanvasEngine }) => {
  const screenRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const bindContainer = (container: HTMLDivElement) => {
    const screen = screenRef.current;
    const overlay = overlayRef.current;
    if (!screen || !overlay) {
      return;
    }

    engine.surface.attach(screen, overlay);

    const syncSize = () => {
      const dpr = globalThis.devicePixelRatio || 1;
      engine.surface.resize(container.clientWidth, container.clientHeight, dpr);
    };

    syncSize();
    // Fit the document into view on first attach, once the viewport is sized.
    if (engine.document.getDocument()) {
      engine.viewport.fitToView();
    }

    const observer = new ResizeObserver(syncSize);
    observer.observe(container);

    return () => {
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
      {/*
       * The text-editing portal: a positioned contenteditable that overlays the
       * canvas whenever a text-edit session is active. It lives inside this
       * relatively-positioned container so its `documentToScreen`-derived
       * absolute offsets share the same origin as the canvas targets.
       */}
      <TextEditPortal engine={engine} />
    </Box>
  );
};

const CANVAS_STYLE: CSSProperties = {
  height: '100%',
  inset: 0,
  position: 'absolute',
  touchAction: 'none',
  width: '100%',
};

const OVERLAY_STYLE: CSSProperties = { ...CANVAS_STYLE, zIndex: 1 };
