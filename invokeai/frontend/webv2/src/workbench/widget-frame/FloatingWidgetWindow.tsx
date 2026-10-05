import type { FloatingWidgetMode, FloatingWidgetState } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { Box, Flex, HStack, Icon, Separator, type SystemStyleObject, Text } from '@chakra-ui/react';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { IconButton } from '@platform/ui/Button';
import { ResizeCorner, trackResizeDrag, usePointerDrag } from '@platform/ui/ResizeHandle';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  clampWindowToViewport,
  commitResizedAxes,
  FLOATING_MIN_HEIGHT_PX,
  FLOATING_MIN_WIDTH_PX,
  FLOATING_VIEWPORT_MARGIN_PX,
  resizeFloatingGeometry,
  type FloatingGeometry,
  type FloatingResizeEdge,
} from '@workbench/floatingWindows';
import { useFloatingWindowFocus, useWorkbenchFocus } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
import { DOCK_DESTINATION_ICONS, resolveDockLabel, resolveWidgetInstanceLabel } from '@workbench/widgetLabels';
import { useActiveProjectId, useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { ChevronsDownUpIcon, ChevronsUpDownIcon, Maximize2Icon, Minimize2Icon, TriangleAlertIcon } from 'lucide-react';
import {
  Activity,
  Component,
  Suspense,
  useCallback,
  useMemo,
  useRef,
  useSyncExternalStore,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';
import { useTranslation } from 'react-i18next';

import { WidgetChromeSlotById, WidgetRendererById } from './WidgetRenderer';
import { areWidgetRenderInstancesEqual } from './widgetRenderInstance';

/** Below Chakra dialogs/popovers/toasts; above the docked shell. */
const FLOATING_BASE_Z_INDEX = 800;
/** Keyboard step for moving and resizing, matching the panel resize handles. */
const FLOATING_STEP_PX = 16;

const MARGIN = `${FLOATING_VIEWPORT_MARGIN_PX}px`;

// Inset, so it shows on a maximized window too and is not clipped at the viewport's edge.
const FOCUS_RING = { outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '-2px' } as const;

/**
 * One static rule set for every window. The stored geometry arrives as custom properties, so committing a move
 * or a raise changes inline values rather than minting a class per geometry.
 *
 * The position keeps a grabbable sliver on screen and the size never exceeds the viewport, both without touching
 * what is stored: a window persisted on a larger display, or left behind by a shrunken browser window, comes back
 * as it was when there is room again (`clampWindowToViewport` is the same policy for what a gesture commits).
 *
 * A gesture previews its geometry in `--fw-preview-*` and names the mode it started in. The preview applies only
 * while the window is still in that mode, so a window that maximizes or shades under a gesture shows its new
 * frame at once.
 */
const WINDOW_SX: SystemStyleObject = {
  '--fw-left': 'var(--fw-x)',
  '--fw-top': 'var(--fw-y)',
  '--fw-width': 'var(--fw-w)',
  '--fw-height': 'var(--fw-h)',
  '&[data-floating-mode="windowed"][data-floating-preview="windowed"], &[data-floating-mode="shaded"][data-floating-preview="shaded"]':
    {
      '--fw-left': 'var(--fw-preview-x, var(--fw-x))',
      '--fw-top': 'var(--fw-preview-y, var(--fw-y))',
      '--fw-width': 'var(--fw-preview-w, var(--fw-w))',
      '--fw-height': 'var(--fw-preview-h, var(--fw-h))',
    },
  height: 'min(var(--fw-height), 100vh)',
  left: `clamp(calc(${MARGIN} - min(var(--fw-width), 100vw)), var(--fw-left), calc(100vw - ${MARGIN}))`,
  top: `clamp(0px, var(--fw-top), calc(100vh - ${MARGIN}))`,
  width: 'min(var(--fw-width), 100vw)',
  '&[data-floating-mode="shaded"]': { height: 'auto' },
  '&[data-floating-mode="maximized"]': { height: '100vh', left: '0', top: '0', width: '100vw' },
  // Keyboard focus that lands on the window itself — after Float, or a marker's reveal — shows on its frame: the
  // frame is its own layer and would cover a ring drawn on the root. It shows whatever the outline preference is.
  outline: 'none',
  '&:focus-visible > [data-floating-frame]': FOCUS_RING,
};

// The title bar has no room for a scrollbar; the strip scrolls by focus, a sideways scroll, and touch.
const ACTIONS_SCROLL_SX: SystemStyleObject = { '&::-webkit-scrollbar': { display: 'none' }, scrollbarWidth: 'none' };

const getViewport = () => ({ height: window.innerHeight, width: window.innerWidth });

const keepInViewport = (geometry: FloatingGeometry): FloatingGeometry => clampWindowToViewport(geometry, getViewport());

/** What resizing the rectangle on screen commits, or null for nothing (see {@link commitResizedAxes}). */
const resizeOnScreen = (
  stored: FloatingGeometry,
  start: FloatingGeometry,
  edge: FloatingResizeEdge,
  deltaX: number,
  deltaY: number
): FloatingGeometry | null => {
  const viewport = getViewport();
  // A window is never shown larger than the viewport, so a resize does not grow past it either.
  const max = { heightPx: viewport.height, widthPx: viewport.width };

  return commitResizedAxes(stored, start, resizeFloatingGeometry(start, edge, deltaX, deltaY, max), viewport);
};

// A drag rewrites the preview properties every frame. Custom properties inherit by default, which would restyle
// the whole widget inside the window each time; registered as non-inheriting, only the window itself is restyled.
// The universal syntax keeps the `var(--fw-preview-x, var(--fw-x))` fallback working while a property is unset.
if (typeof CSS !== 'undefined' && typeof CSS.registerProperty === 'function') {
  for (const name of [
    'x',
    'y',
    'w',
    'h',
    'preview-x',
    'preview-y',
    'preview-w',
    'preview-h',
    'left',
    'top',
    'width',
    'height',
  ]) {
    try {
      CSS.registerProperty({ inherits: false, name: `--fw-${name}`, syntax: '*' });
    } catch {
      // Already registered: this module was evaluated before (hot reload).
    }
  }
}

const PREVIEW_PROPERTIES = {
  heightPx: '--fw-preview-h',
  widthPx: '--fw-preview-w',
  x: '--fw-preview-x',
  y: '--fw-preview-y',
} as const satisfies Record<keyof FloatingGeometry, string>;

/** How far a resize handle reaches to either side of the window's border. */
const HANDLE_REACH = '4px';
/** The corner squares' side — the labelled grip's size — so the edge strips run between them without overlap. */
const HANDLE_CORNER = '1rem';
const EDGE_INSET = `calc(${HANDLE_CORNER} - ${HANDLE_REACH})`;
const HANDLE_OFFSET = `-${HANDLE_REACH}`;
const EDGE_THICKNESS = `calc(${HANDLE_REACH} * 2)`;
const OUTER_CORNER = `calc(${HANDLE_CORNER} + ${HANDLE_REACH})`;

/**
 * The pointer-only resize handles: four edges and four corners. The bottom-right corner's inside is the labelled
 * `ResizeCorner`, the one handle that also takes the keyboard, so the strips beside it stop at its square and the
 * pointer-only handle there is clipped to the band outside the border.
 */
const RESIZE_HANDLES: readonly { cursor: string; edge: FloatingResizeEdge; sx: SystemStyleObject }[] = [
  {
    cursor: 'ns-resize',
    edge: 'n',
    sx: { height: EDGE_THICKNESS, left: EDGE_INSET, right: EDGE_INSET, top: HANDLE_OFFSET },
  },
  {
    cursor: 'ns-resize',
    edge: 's',
    sx: { bottom: HANDLE_OFFSET, height: EDGE_THICKNESS, left: EDGE_INSET, right: HANDLE_CORNER },
  },
  {
    cursor: 'ew-resize',
    edge: 'w',
    sx: { bottom: EDGE_INSET, left: HANDLE_OFFSET, top: EDGE_INSET, width: EDGE_THICKNESS },
  },
  {
    cursor: 'ew-resize',
    edge: 'e',
    sx: { bottom: HANDLE_CORNER, right: HANDLE_OFFSET, top: EDGE_INSET, width: EDGE_THICKNESS },
  },
  {
    cursor: 'nwse-resize',
    edge: 'nw',
    sx: { height: HANDLE_CORNER, left: HANDLE_OFFSET, top: HANDLE_OFFSET, width: HANDLE_CORNER },
  },
  {
    cursor: 'nesw-resize',
    edge: 'ne',
    sx: { height: HANDLE_CORNER, right: HANDLE_OFFSET, top: HANDLE_OFFSET, width: HANDLE_CORNER },
  },
  {
    cursor: 'nesw-resize',
    edge: 'sw',
    sx: { bottom: HANDLE_OFFSET, height: HANDLE_CORNER, left: HANDLE_OFFSET, width: HANDLE_CORNER },
  },
  {
    cursor: 'nwse-resize',
    edge: 'se',
    sx: {
      bottom: HANDLE_OFFSET,
      // An L around the grip: the box minus the grip's square, which is the part inside the border.
      clipPath: `polygon(${HANDLE_CORNER} 0, 100% 0, 100% 100%, 0 100%, 0 ${HANDLE_CORNER}, ${HANDLE_CORNER} ${HANDLE_CORNER})`,
      height: OUTER_CORNER,
      right: HANDLE_OFFSET,
      width: OUTER_CORNER,
    },
  },
];

const FloatingResizeHandles = ({
  onResizeStart,
}: {
  onResizeStart: (event: ReactPointerEvent<HTMLDivElement>, edge: FloatingResizeEdge, cursor: string) => void;
}) => {
  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const handle = RESIZE_HANDLES.find(({ edge }) => edge === event.currentTarget.dataset.resizeEdge);

      if (handle) {
        onResizeStart(event, handle.edge, handle.cursor);
      }
    },
    [onResizeStart]
  );

  return (
    <>
      {RESIZE_HANDLES.map(({ cursor, edge, sx }) => (
        <Box
          key={edge}
          aria-hidden
          css={sx}
          cursor={cursor}
          data-resize-edge={edge}
          position="absolute"
          // `preventDefault` on pointerdown does not stop touch panning.
          touchAction="none"
          zIndex="1"
          onPointerDown={handlePointerDown}
        />
      ))}
    </>
  );
};

/**
 * One pointer gesture on the window. Everything it needs from the moment it began is here, so nothing it does
 * later depends on which render its handlers were created in.
 */
interface WindowGesture {
  edge: FloatingResizeEdge | 'move';
  /** What a release would commit; null until the pointer has actually moved. */
  live: FloatingGeometry | null;
  mode: FloatingWidgetMode;
  session: AbortController | null;
  /** Where the window was on screen. */
  start: FloatingGeometry;
  /** What was stored, which the gesture keeps on every axis it does not change. */
  stored: FloatingGeometry;
}

const subscribeToViewportSize = (onChange: () => void): (() => void) => {
  window.addEventListener('resize', onChange);
  return () => window.removeEventListener('resize', onChange);
};
const getViewportWidthPx = (): number => window.innerWidth;
const getViewportHeightPx = (): number => window.innerHeight;

/**
 * The window's one labelled resize control. It announces the size on screen, which the viewport may cap below the
 * stored one, and follows the viewport: a browser window that shrinks re-renders this grip and nothing else, so
 * it never announces a size nobody sees.
 */
const FloatingResizeCorner = ({
  heightPx,
  onKeyDown,
  onPointerDown,
  widthPx,
}: {
  /** The stored size. */
  heightPx: number;
  widthPx: number;
  onKeyDown: (event: ReactKeyboardEvent<HTMLDivElement>) => void;
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
}) => {
  const { t } = useTranslation();
  const viewportWidthPx = useSyncExternalStore(subscribeToViewportSize, getViewportWidthPx);
  const viewportHeightPx = useSyncExternalStore(subscribeToViewportSize, getViewportHeightPx);
  const shownWidthPx = Math.min(widthPx, viewportWidthPx);
  const shownHeightPx = Math.min(heightPx, viewportHeightPx);

  return (
    <ResizeCorner
      label={t('widgets.floating.resize')}
      valueMax={viewportWidthPx}
      // A viewport narrower than the minimum shows the window narrower still.
      valueMin={Math.min(FLOATING_MIN_WIDTH_PX, shownWidthPx)}
      valueNow={shownWidthPx}
      valueText={t('widgets.floating.resizeValue', { height: shownHeightPx, width: shownWidthPx })}
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
    />
  );
};

/**
 * Isolate arbitrary widget controls from title-bar drag and double-click maximize gestures; not every control is a
 * button.
 */
const stopChromeEvent = (event: ReactPointerEvent<HTMLDivElement> | ReactMouseEvent<HTMLDivElement>): void =>
  event.stopPropagation();

/**
 * Contain failed widget chrome so the window and dock control survive. Body retry does not reset this boundary;
 * chrome returns on the next window mount.
 */
class FloatingChromeBoundary extends Component<{ children: ReactNode }, { hasFailed: boolean }> {
  state = { hasFailed: false };

  static getDerivedStateFromError(): { hasFailed: boolean } {
    return { hasFailed: true };
  }

  render() {
    return this.state.hasFailed ? null : this.props.children;
  }
}

export const FloatingWidgetWindow = ({
  instanceId,
  stackRank,
  state,
}: {
  instanceId: WidgetInstanceId;
  /** 0-based position in the layer's stacking order (0 = bottom window). */
  stackRank: number;
  state: FloatingWidgetState;
}) => {
  const { t } = useTranslation();
  const { widgets } = useWorkbenchCommands();
  const { getWidgetById } = useWorkbenchWidgetRegistry();
  const instance = useActiveProjectSelector(
    (project) => project.widgetInstances[instanceId],
    areWidgetRenderInstancesEqual
  );
  const projectId = useActiveProjectId();
  const { activate, isHighlighted } = useFloatingWindowFocus(instanceId, projectId);
  const { focusRegion } = useWorkbenchFocus();
  const windowRef = useRef<HTMLDivElement>(null);
  const gestureRef = useRef<WindowGesture | null>(null);
  const startDrag = usePointerDrag();

  const widget = instance ? getWidgetById(instance.typeId) : undefined;
  const { heightPx, mode, widthPx, x, y } = state;

  const commitGeometry = useCallback(
    // The rendered rectangle a gesture starts from can sit on fractional pixels.
    (geometry: FloatingGeometry) =>
      widgets.setFloatingGeometry(instanceId, {
        heightPx: Math.round(geometry.heightPx),
        widthPx: Math.round(geometry.widthPx),
        x: Math.round(geometry.x),
        y: Math.round(geometry.y),
      }),
    [instanceId, widgets]
  );

  /**
   * Where the window is on screen right now. Gestures and key steps start here, not from the stored geometry:
   * CSS may be holding the window inside the viewport or capping its size, and starting from the stored values
   * would make it jump.
   */
  const readRenderedGeometry = useCallback((): FloatingGeometry | null => {
    const rect = windowRef.current?.getBoundingClientRect();

    return rect ? { heightPx: rect.height, widthPx: rect.width, x: rect.left, y: rect.top } : null;
  }, []);

  // A gesture writes its geometry as inline custom properties so it renders without React; the committed render
  // then replaces it. The mode it names is what limits the preview to the frame the gesture began in.
  const writePreview = useCallback((preview: { geometry: FloatingGeometry; mode: FloatingWidgetMode } | null) => {
    const element = windowRef.current;

    if (!element) {
      return;
    }

    for (const key of Object.keys(PREVIEW_PROPERTIES) as (keyof FloatingGeometry)[]) {
      if (preview) {
        element.style.setProperty(PREVIEW_PROPERTIES[key], `${preview.geometry[key]}px`);
      } else {
        element.style.removeProperty(PREVIEW_PROPERTIES[key]);
      }
    }
    if (preview) {
      element.dataset.floatingPreview = preview.mode;
    } else {
      delete element.dataset.floatingPreview;
    }
  }, []);

  const cancelGesture = useCallback(() => {
    const gesture = gestureRef.current;

    gestureRef.current = null;
    writePreview(null);
    // Ends the pointer session too: the drag cursor and the Escape capture do not outlive the gesture.
    gesture?.session?.abort();
  }, [writePreview]);

  /** Every handle and the title bar start here, so one gesture owns the window at a time. */
  const startWindowDrag = useCallback(
    (event: ReactPointerEvent<HTMLElement>, edge: WindowGesture['edge'], cursor: string) => {
      if (event.button !== 0) {
        return;
      }

      // One gesture at a time. This also drops a committed preview still waiting out its last frame, so the
      // starting rectangle is read from the committed render.
      cancelGesture();

      const handle = event.currentTarget;
      const start = readRenderedGeometry();

      if (!start) {
        return;
      }

      const stored = { heightPx, widthPx, x, y };
      const gesture: WindowGesture = { edge, live: null, mode, session: null, start, stored };
      const isCurrent = () => gestureRef.current === gesture;

      gestureRef.current = gesture;
      gesture.session = startDrag(event, {
        cursor,
        onEnd: (reason) => {
          if (!isCurrent()) {
            return;
          }

          gestureRef.current = null;
          // Nothing moved, the user backed out, or the window is no longer in the mode the offsets were for.
          if (!gesture.live || reason === 'escape' || windowRef.current?.dataset.floatingMode !== gesture.mode) {
            writePreview(null);
            return;
          }

          // Hold the preview one more frame so the committed render replaces it without a flash; scheduled first
          // so a throwing commit still clears it. A gesture that starts before that frame is safe from it: frames
          // run in order, and a gesture writes its first preview in a later one.
          requestAnimationFrame(() => writePreview(null));
          commitGeometry(gesture.live);
        },
        onMove: (deltaX, deltaY) => {
          if (!isCurrent()) {
            return;
          }
          // The window changed mode under the gesture — a shortcut, an undo, a preset. Its offsets no longer
          // describe anything on screen, so it stands down and commits nothing.
          if (windowRef.current?.dataset.floatingMode !== gesture.mode) {
            cancelGesture();
            return;
          }
          // A gesture commits only what it changed. A press that never moves, or a drag against the minimum or
          // the viewport's cap, changes nothing and commits nothing; a move keeps the stored size and a resize
          // keeps the axis it does not touch. What is on screen there may only be the viewport's clamp or cap,
          // which is not the user's choice to persist.
          gesture.live =
            edge !== 'move'
              ? resizeOnScreen(stored, start, edge, deltaX, deltaY)
              : deltaX === 0 && deltaY === 0
                ? null
                : keepInViewport({
                    heightPx: stored.heightPx,
                    widthPx: stored.widthPx,
                    x: start.x + deltaX,
                    y: start.y + deltaY,
                  });
          writePreview(gesture.live ? { geometry: gesture.live, mode: gesture.mode } : null);
        },
      });

      const { signal } = gesture.session;

      handle.setAttribute('data-dragging', '');
      if (edge !== 'move') {
        trackResizeDrag(signal);
      }
      // The session aborts when it ends, when the window unmounts, and when another gesture replaces it; only a
      // gesture that never ended still has a preview to undo.
      signal.addEventListener('abort', () => {
        handle.removeAttribute('data-dragging');
        if (isCurrent()) {
          gestureRef.current = null;
          writePreview(null);
        }
      });
    },
    [cancelGesture, commitGeometry, heightPx, mode, readRenderedGeometry, startDrag, widthPx, writePreview, x, y]
  );
  const handleTitlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (mode !== 'maximized' && !(event.target as HTMLElement).closest('button')) {
        startWindowDrag(event, 'move', 'move');
      }
    },
    [mode, startWindowDrag]
  );
  const handleCornerPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => startWindowDrag(event, 'se', 'nwse-resize'),
    [startWindowDrag]
  );

  // Provide keyboard move/resize alongside shade/maximize/dock, matching panel resize steps.
  const handleTitleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const step = event.shiftKey ? FLOATING_STEP_PX * 2 : FLOATING_STEP_PX;
      const offsets: Partial<Record<string, [number, number]>> = {
        ArrowDown: [0, step],
        ArrowLeft: [-step, 0],
        ArrowRight: [step, 0],
        ArrowUp: [0, -step],
      };
      const offset = offsets[event.key];

      // Handle movement keys only on the bar itself; controls' bubbling arrows must not alter geometry.
      if (!offset || mode === 'maximized' || event.target !== event.currentTarget) {
        return;
      }

      const rendered = readRenderedGeometry();

      if (!rendered) {
        return;
      }

      event.preventDefault();
      // Like a pointer move: from where the window is on screen, keeping the stored size.
      commitGeometry(keepInViewport({ heightPx, widthPx, x: rendered.x + offset[0], y: rendered.y + offset[1] }));
    },
    [commitGeometry, heightPx, mode, readRenderedGeometry, widthPx]
  );

  const handleResizeKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const rendered = readRenderedGeometry();

      if (!rendered) {
        return;
      }

      const step = event.shiftKey ? FLOATING_STEP_PX * 2 : FLOATING_STEP_PX;
      const offsets: Partial<Record<string, [number, number]>> = {
        ArrowDown: [0, step],
        ArrowLeft: [-step, 0],
        ArrowRight: [step, 0],
        ArrowUp: [0, -step],
        Home: [FLOATING_MIN_WIDTH_PX - rendered.widthPx, FLOATING_MIN_HEIGHT_PX - rendered.heightPx],
      };
      const offset = offsets[event.key];

      if (!offset) {
        return;
      }

      event.preventDefault();

      const resized = resizeOnScreen({ heightPx, widthPx, x, y }, rendered, 'se', offset[0], offset[1]);

      if (resized) {
        commitGeometry(resized);
      }
    },
    [commitGeometry, heightPx, readRenderedGeometry, widthPx, x, y]
  );

  // Pointer-down and keyboard focus make this the active window and raise it; hover does neither. A late event
  // from a project that has left the screen is refused before it can raise anything.
  //
  // Focus raises only when it makes this window the active one. Focus moving around inside the window that is
  // already active — a Tab, a closing dialog or popover handing focus back — writes nothing, and cannot re-bury a
  // window that a recall just revealed on top of it. A press always asks; that is a no-op for the topmost window.
  const handleFocusCapture = useCallback(() => {
    if (activate() === 'activated') {
      widgets.raiseFloating(instanceId);
    }
  }, [activate, instanceId, widgets]);
  // A press on content that takes no focus of its own (or on the title bar, whose drag prevents it) would leave
  // keyboard focus wherever it was, and the keys would go there while this window shows as active. Presses that
  // reach here from a menu or popover portaled out of the window keep their own focus.
  const handlePointerDownCapture = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const element = event.currentTarget;

      if (activate({ byPointer: true }) !== 'refused') {
        widgets.raiseFloating(instanceId);
      }
      if (event.target instanceof Node && element.contains(event.target) && !element.contains(document.activeElement)) {
        element.focus({ preventScroll: true });
      }
    },
    [activate, instanceId, widgets]
  );
  const typeId = instance?.typeId;
  // Flush drafts before docking remounts the widget; registry cleanup only removes flushers. Focus follows the
  // widget to the region it returns to.
  const handleDock = useCallback(() => {
    flushWorkbenchDrafts();
    widgets.dockFloating(instanceId);
    if (typeId) {
      focusRegion(state.returnRegion, typeId);
    }
  }, [focusRegion, instanceId, state.returnRegion, typeId, widgets]);
  const handleToggleShade = useCallback(() => {
    // Shading hides the body: its editors keep their state but stop running, so their drafts are saved first.
    if (mode !== 'shaded') {
      flushWorkbenchDrafts();
    }
    widgets.setFloatingMode(instanceId, mode === 'shaded' ? 'windowed' : 'shaded');
  }, [instanceId, mode, widgets]);
  // Maximizing leaves the windowed geometry untouched, so Restore returns to exactly it.
  const handleToggleMaximize = useCallback(
    () => widgets.setFloatingMode(instanceId, mode === 'maximized' ? 'windowed' : 'maximized'),
    [instanceId, mode, widgets]
  );
  const handleTitleDoubleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      // A double-click on a title-bar button must not also change the window's mode.
      if ((event.target as HTMLElement).closest('button')) {
        return;
      }
      // A collapsed window opens back up; jumping straight from a title bar to the whole viewport would surprise.
      if (mode === 'shaded') {
        widgets.setFloatingMode(instanceId, 'windowed');
        return;
      }

      handleToggleMaximize();
    },
    [handleToggleMaximize, instanceId, mode, widgets]
  );

  // The stored geometry and stacking as inline values of the one static rule set.
  const geometryStyle = useMemo(
    () =>
      ({
        '--fw-h': `${heightPx}px`,
        '--fw-w': `${widthPx}px`,
        '--fw-x': `${x}px`,
        '--fw-y': `${y}px`,
        zIndex: FLOATING_BASE_Z_INDEX + stackRank,
      }) as CSSProperties,
    [heightPx, stackRank, widthPx, x, y]
  );

  if (!instance) {
    return null;
  }

  // Retain window chrome for missing or failed widgets so users can dock them back to the retry surface.
  const isEnabled = widget?.status === 'enabled';
  const label = widget ? resolveWidgetInstanceLabel(instance, widget.manifest, t) : (instance.title ?? instance.id);
  const dockLabel = resolveDockLabel(state.returnRegion, t);
  const isMaximized = mode === 'maximized';
  const isShaded = mode === 'shaded';

  return (
    <Box
      ref={windowRef}
      aria-label={label}
      css={WINDOW_SX}
      position="fixed"
      role="group"
      // Focusable by script only, so focus can move into a window that has just been floated or revealed.
      style={geometryStyle}
      tabIndex={-1}
      data-floating-mode={mode}
      data-floating-window={instanceId}
      data-highlighted={isHighlighted}
      // Mark floating widget identity and region so hotkeys target it rather than the last focused docked widget.
      data-hotkey-widget-instance-id={instanceId}
      data-hotkey-widget-region="floating"
      data-hotkey-widget-type-id={instance.typeId}
      onFocusCapture={handleFocusCapture}
      onPointerDownCapture={handlePointerDownCapture}
    >
      <Flex
        bg="bg.subtle"
        // The active window carries the same accent outline, under the same preference, as a focused region.
        borderColor={isHighlighted ? 'accent.solid' : 'border.emphasized'}
        borderWidth="1px"
        direction="column"
        h="full"
        // Its own stacking context: nothing a widget stacks inside can rise above the resize handles.
        isolation="isolate"
        overflow="hidden"
        data-floating-frame=""
        rounded={isMaximized ? 'none' : 'md'}
        shadow="xl"
        transition="border-color var(--wb-motion-duration-fast) ease"
      >
        <HStack
          aria-label={t('widgets.floating.move', { label })}
          borderBottomWidth={isShaded ? 0 : '1px'}
          cursor={isMaximized ? 'default' : 'move'}
          flexShrink={0}
          gap="1.5"
          // Tighter than a docked panel's header: a window's chrome should take as little of it as it can. The
          // end padding keeps the Dock button clear of the top-right resize corner, which reaches 12px inside.
          h={8}
          pe="3"
          ps="3"
          tabIndex={isMaximized ? undefined : 0}
          // `preventDefault` on pointerdown does not stop touch panning: without
          // this the browser claims the gesture and cancels the drag.
          touchAction="none"
          userSelect="none"
          _focusVisible={FOCUS_RING}
          outline="none"
          onDoubleClick={handleTitleDoubleClick}
          onKeyDown={handleTitleKeyDown}
          onPointerDown={handleTitlePointerDown}
        >
          {/*
           * The label takes only the room the controls leave, so a narrow window truncates it first — down to its
           * icon, which keeps its place rather than sliding over the actions.
           */}
          <HStack flex="1 1 0" gap="1.5" minW="4" overflow="hidden">
            {widget ? <WidgetIcon boxSize="4" flexShrink={0} icon={widget.manifest.icon} /> : null}
            <Text fontSize="md" fontWeight="700" truncate>
              {label}
            </Text>
          </HStack>
          {/*
           * Render widget actions and settings in a row because floating content has no frame header; window
           * controls already own layout actions. Contributed actions give way before they can push the window's
           * own controls out of a narrow title bar: the ones that do not fit scroll, so focus brings each into
           * view and a sideways scroll or swipe reaches the rest.
           */}
          {isEnabled && widget ? (
            <FloatingChromeBoundary>
              <Suspense fallback={null}>
                <HStack
                  css={ACTIONS_SCROLL_SX}
                  flex="0 1 auto"
                  gap="1"
                  // Room for a focused action's ring inside the scrollport, kept when focus scrolls one into view.
                  m="-1"
                  minW="0"
                  overflowX="auto"
                  overflowY="hidden"
                  // A swipe on the strip scrolls it and nothing behind it, even at either end.
                  overscrollBehaviorX="contain"
                  p="1"
                  scrollPaddingInline="1"
                  touchAction="pan-x"
                  data-floating-actions=""
                  onDoubleClick={stopChromeEvent}
                  onPointerDown={stopChromeEvent}
                >
                  <WidgetChromeSlotById instanceId={instanceId} region="floating" slot="viewActions" widget={widget} />
                </HStack>
              </Suspense>
            </FloatingChromeBoundary>
          ) : null}
          <HStack flexShrink={0} gap="1" data-floating-controls="">
            {isEnabled && widget ? <Separator h="4" mx="0.5" orientation="vertical" /> : null}
            <Tooltip content={isShaded ? t('widgets.floating.unshade') : t('widgets.floating.shade')}>
              <IconButton
                aria-label={isShaded ? t('widgets.floating.unshade') : t('widgets.floating.shade')}
                color="fg.muted"
                size="sm"
                variant="ghost"
                onClick={handleToggleShade}
              >
                <Icon as={isShaded ? ChevronsUpDownIcon : ChevronsDownUpIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
            <Tooltip content={isMaximized ? t('widgets.floating.restore') : t('widgets.floating.maximize')}>
              <IconButton
                aria-label={isMaximized ? t('widgets.floating.restore') : t('widgets.floating.maximize')}
                color="fg.muted"
                size="sm"
                variant="ghost"
                onClick={handleToggleMaximize}
              >
                <Icon as={isMaximized ? Minimize2Icon : Maximize2Icon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
            <Tooltip content={dockLabel}>
              <IconButton aria-label={dockLabel} color="fg.muted" size="sm" variant="ghost" onClick={handleDock}>
                <Icon as={DOCK_DESTINATION_ICONS[state.returnRegion]} boxSize="3.5" />
              </IconButton>
            </Tooltip>
          </HStack>
        </HStack>
        {/* A shaded body is hidden, not unmounted: its local state survives, and no second copy mounts. */}
        <Activity mode={isShaded ? 'hidden' : 'visible'}>
          <Flex direction="column" flex="1" minH="0" overflow="hidden">
            {isEnabled && widget ? (
              <WidgetRendererById instanceId={instance.id} region="floating" widget={widget} />
            ) : (
              <HStack color="fg.error" gap="1.5" p="3">
                <Icon as={TriangleAlertIcon} boxSize="3.5" />
                <Text fontSize="md">{t('widgets.failure.title', { label })}</Text>
              </HStack>
            )}
          </Flex>
        </Activity>
      </Flex>
      {isShaded || isMaximized ? null : (
        <>
          <FloatingResizeHandles onResizeStart={startWindowDrag} />
          <FloatingResizeCorner
            heightPx={heightPx}
            widthPx={widthPx}
            onKeyDown={handleResizeKeyDown}
            onPointerDown={handleCornerPointerDown}
          />
        </>
      )}
    </Box>
  );
};
