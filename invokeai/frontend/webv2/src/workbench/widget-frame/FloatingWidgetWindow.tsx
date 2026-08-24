import type { FloatingWidgetState } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { Box, Flex, HStack, Icon, Text } from '@chakra-ui/react';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { IconButton, Tooltip } from '@platform/ui';
import {
  clampWindowToViewport,
  FLOATING_MIN_HEIGHT_PX,
  FLOATING_MIN_WIDTH_PX,
  type FloatingGeometry,
} from '@workbench/floatingWindows';
import { WidgetIcon } from '@workbench/iconResolver';
import { resolveWidgetInstanceLabel } from '@workbench/widgetLabels';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import {
  ChevronsDownUpIcon,
  ChevronsUpDownIcon,
  Maximize2Icon,
  Minimize2Icon,
  PanelRightIcon,
  TriangleAlertIcon,
} from 'lucide-react';
import {
  Component,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
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

/**
 * The title bar is a drag handle first and a control strip second: a
 * `pointerdown` anywhere on it that is not inside a `<button>` starts a window
 * drag, and a double-click shades it. Widget-supplied chrome is arbitrary —
 * a switch, a slider, a menu trigger rendered as a div — so it is isolated
 * from both gestures rather than trusted to be a button.
 */
const stopChromeEvent = (event: ReactPointerEvent<HTMLDivElement> | ReactMouseEvent<HTMLDivElement>): void =>
  event.stopPropagation();

/**
 * Drops the widget's title-bar chrome if it throws, keeping the window itself
 * alive. The chunk carrying a widget's implementation can fail to load — a
 * deploy replaced the hashed file, or the tab is offline — and the deferred
 * resource then hands `use()` the same rejected thenable on every render.
 * Inside the body that throw lands in `WidgetFailureBoundary`, which offers a
 * retry; up here there is no boundary between this bar and the app root, so an
 * unguarded throw would take the whole workbench down — and with it the dock
 * control that is a floated widget's only way back to the rail.
 *
 * Recovery is deliberately one-way: a retry from the body's failure card
 * reloads the implementation but does not reset this boundary, so the chrome
 * returns on the window's next mount. Chrome that stays missing is a far
 * cheaper failure than chrome that cannot be reached at all.
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

/**
 * One detached widget window: fixed-position chrome with a draggable title
 * bar, a corner resize handle, shade/maximize/dock controls, and the standard
 * widget renderer as its body. Drag and resize use the same raw-pointer
 * pattern as the panel resize handles — transient local px state, one commit
 * to the reducer on release.
 */
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
  const [dragGeometry, setDragGeometry] = useState<FloatingGeometry | null>(null);

  const widget = instance ? getWidgetById(instance.typeId) : undefined;

  const commitGeometry = useCallback(
    (geometry: FloatingGeometry) => {
      const clamped = clampWindowToViewport(geometry, { height: window.innerHeight, width: window.innerWidth });
      widgets.setFloatingGeometry(instanceId, clamped);
    },
    [instanceId, widgets]
  );

  const pointerSessionRef = useRef<AbortController | null>(null);

  // A drag can outlive the window — docking from a command, an applied preset,
  // or a project switch all unmount mid-gesture — and window-level listeners
  // would then stay bound for the rest of the session.
  useEffect(() => () => pointerSessionRef.current?.abort(), []);

  const beginPointerOperation = useCallback(
    (
      event: ReactPointerEvent<HTMLDivElement>,
      apply: (deltaX: number, deltaY: number, start: FloatingGeometry) => FloatingGeometry
    ) => {
      event.preventDefault();
      // Capture keeps the gesture addressed to this window even when the
      // pointer crosses an iframe or another window's chrome.
      event.currentTarget.setPointerCapture(event.pointerId);

      const startX = event.clientX;
      const startY = event.clientY;
      const start: FloatingGeometry = { heightPx: state.heightPx, widthPx: state.widthPx, x: state.x, y: state.y };
      let next = start;
      const pointerSession = new AbortController();

      pointerSessionRef.current?.abort();
      pointerSessionRef.current = pointerSession;

      const handlePointerUp = () => {
        pointerSession.abort();
        pointerSessionRef.current = null;
        setDragGeometry(null);
        commitGeometry(next);
      };

      const handlePointerMove = (moveEvent: PointerEvent) => {
        // Releasing the button over another application swallows `pointerup`,
        // and the window would then follow the cursor with nothing held. The
        // first move that arrives with no button down ends the drag instead.
        if (moveEvent.buttons === 0) {
          handlePointerUp();

          return;
        }

        next = apply(moveEvent.clientX - startX, moveEvent.clientY - startY, start);
        setDragGeometry(next);
      };

      window.addEventListener('pointermove', handlePointerMove, { signal: pointerSession.signal });
      window.addEventListener('pointerup', handlePointerUp, { signal: pointerSession.signal });
      window.addEventListener('pointercancel', handlePointerUp, { signal: pointerSession.signal });
    },
    [commitGeometry, state.heightPx, state.widthPx, state.x, state.y]
  );

  const handleTitlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0 || state.mode === 'maximized' || (event.target as HTMLElement).closest('button')) {
        return;
      }

      beginPointerOperation(event, (deltaX, deltaY, start) => ({ ...start, x: start.x + deltaX, y: start.y + deltaY }));
    },
    [beginPointerOperation, state.mode]
  );

  const handleResizePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return;
      }

      beginPointerOperation(event, (deltaX, deltaY, start) => ({
        ...start,
        heightPx: Math.max(FLOATING_MIN_HEIGHT_PX, start.heightPx + deltaY),
        widthPx: Math.max(FLOATING_MIN_WIDTH_PX, start.widthPx + deltaX),
      }));
    },
    [beginPointerOperation]
  );

  // Pointer gestures are not the only way to place a window: without these the
  // keyboard can shade, maximize and dock a floated widget but never move or
  // resize it. Stepping mirrors the panel resize handles in `WidgetFrames`.
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

      // Only the bar itself moves the window. It hosts real controls — its own
      // shade/maximize/dock buttons and the widget's header actions — and
      // React sees their keystrokes bubble up here: without this, an arrow key
      // pressed on a focused toggle moved the window 16px and wrote the new
      // geometry to the reducer.
      if (!offset || state.mode === 'maximized' || event.target !== event.currentTarget) {
        return;
      }

      event.preventDefault();
      commitGeometry({ ...state, x: state.x + offset[0], y: state.y + offset[1] });
    },
    [commitGeometry, state]
  );

  const handleResizeKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const step = event.shiftKey ? FLOATING_STEP_PX * 2 : FLOATING_STEP_PX;
      const offsets: Partial<Record<string, [number, number]>> = {
        ArrowDown: [0, step],
        ArrowLeft: [-step, 0],
        ArrowRight: [step, 0],
        ArrowUp: [0, -step],
        Home: [FLOATING_MIN_WIDTH_PX - state.widthPx, FLOATING_MIN_HEIGHT_PX - state.heightPx],
      };
      const offset = offsets[event.key];

      if (!offset) {
        return;
      }

      event.preventDefault();
      commitGeometry({
        ...state,
        heightPx: Math.max(FLOATING_MIN_HEIGHT_PX, state.heightPx + offset[1]),
        widthPx: Math.max(FLOATING_MIN_WIDTH_PX, state.widthPx + offset[0]),
      });
    },
    [commitGeometry, state]
  );

  const handleFocus = useCallback(() => widgets.focusFloating(instanceId), [instanceId, widgets]);
  // Docking remounts the widget in its rail. The draft registry's cleanup only
  // deregisters the flusher, so an uncommitted edit needs committing first —
  // the same reason `closeWidgetPlacement` flushes before it unmounts.
  const handleDock = useCallback(() => {
    flushWorkbenchDrafts();
    widgets.dockFloating(instanceId);
  }, [instanceId, widgets]);
  const handleToggleShade = useCallback(
    () => widgets.setFloatingMode(instanceId, state.mode === 'shaded' ? 'windowed' : 'shaded'),
    [instanceId, state.mode, widgets]
  );
  const handleToggleMaximize = useCallback(
    () => widgets.setFloatingMode(instanceId, state.mode === 'maximized' ? 'windowed' : 'maximized'),
    [instanceId, state.mode, widgets]
  );
  const handleTitleDoubleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      // A double-click on a title-bar button (e.g. Maximize) must not also
      // shade the window.
      if (state.mode !== 'maximized' && !(event.target as HTMLElement).closest('button')) {
        handleToggleShade();
      }
    },
    [handleToggleShade, state.mode]
  );

  if (!instance) {
    return null;
  }

  // A floated widget that fails registration — or whose type has gone from the
  // registry entirely in a later build — keeps its chrome. Rendering nothing
  // would strand the instance: it is in no region, so the dock control in this
  // title bar is the only way back to the rail, and to the docked failure card
  // that owns the retry. This is why a `hidden` widget still shows a window
  // here while `getWidgetsForRegion` keeps it out of the rails: a rail the
  // widget is missing from is merely tidy, a window it is missing from is a
  // widget the person cannot reach.
  const isEnabled = widget?.status === 'enabled';
  const label = widget ? resolveWidgetInstanceLabel(instance, widget.manifest, t) : (instance.title ?? instance.id);
  const geometry = dragGeometry ?? state;
  const isMaximized = state.mode === 'maximized';
  const isShaded = state.mode === 'shaded';
  // CSS clamp keeps a grabbable sliver on-screen even for geometry persisted
  // on a larger display (or after the browser window shrinks) — the commit
  // clamp only covers drags on the current viewport.
  const positionProps = isMaximized
    ? { h: '100vh', left: 0, top: 0, w: '100vw' }
    : {
        h: isShaded ? 'auto' : `${geometry.heightPx}px`,
        left: `clamp(${48 - geometry.widthPx}px, ${geometry.x}px, calc(100vw - 48px))`,
        top: `clamp(0px, ${geometry.y}px, calc(100vh - 48px))`,
        w: `${geometry.widthPx}px`,
      };

  return (
    <Flex
      bg="bg.subtle"
      borderColor="border.emphasized"
      borderWidth="1px"
      direction="column"
      overflow="hidden"
      position="fixed"
      rounded={isMaximized ? 'none' : 'md'}
      shadow="xl"
      zIndex={FLOATING_BASE_Z_INDEX + stackRank}
      // The docked frames carry these and the hotkey runtime reads them to tell
      // which widget a keystroke is for. Floating content renders bare — this
      // window is its chrome — so without them `getHotkeyTargetWidget` found
      // nothing and the runtime fell back to the last focused REGION's active
      // widget: Delete pressed over a floating window ran the docked Gallery's
      // delete-selection on whatever that had selected. `floating` is a legal
      // contribution-source region and is what the runtime registers this
      // widget's own contributions under, so its hotkeys now resolve as well —
      // previously they could not fire at all.
      data-hotkey-widget-instance-id={instanceId}
      data-hotkey-widget-region="floating"
      data-hotkey-widget-type-id={instance.typeId}
      onPointerDownCapture={handleFocus}
      {...positionProps}
    >
      <HStack
        aria-label={t('widgets.floating.move', { label })}
        borderBottomWidth={isShaded ? 0 : '1px'}
        cursor={isMaximized ? 'default' : 'move'}
        flexShrink={0}
        gap="1.5"
        h={10}
        justify="space-between"
        pe="2"
        ps="3"
        tabIndex={isMaximized ? undefined : 0}
        // `preventDefault` on pointerdown does not stop touch panning: without
        // this the browser claims the gesture and cancels the drag.
        touchAction="none"
        userSelect="none"
        onDoubleClick={handleTitleDoubleClick}
        onKeyDown={handleTitleKeyDown}
        onPointerDown={handleTitlePointerDown}
      >
        <HStack flex="1" gap="1.5" minW="0">
          {widget ? <WidgetIcon boxSize="4" icon={widget.manifest.icon} /> : null}
          <Text fontSize="xs" fontWeight="700" truncate>
            {label}
          </Text>
        </HStack>
        <HStack flexShrink={0} gap="1">
          {/* The widget's own header toggles: floated content renders bare, so
              without this the docked header's controls would simply vanish on
              float. Frame-level actions stay out — this bar carries its own. */}
          {isEnabled && widget ? (
            <FloatingChromeBoundary>
              <Suspense fallback={null}>
                <Box onDoubleClick={stopChromeEvent} onPointerDown={stopChromeEvent}>
                  <WidgetChromeSlotById instanceId={instanceId} region="floating" slot="viewActions" widget={widget} />
                </Box>
              </Suspense>
            </FloatingChromeBoundary>
          ) : null}
          <Tooltip content={isShaded ? t('widgets.floating.unshade') : t('widgets.floating.shade')}>
            <IconButton
              aria-label={isShaded ? t('widgets.floating.unshade') : t('widgets.floating.shade')}
              color="fg.muted"
              size="2xs"
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
              size="2xs"
              variant="ghost"
              onClick={handleToggleMaximize}
            >
              <Icon as={isMaximized ? Minimize2Icon : Maximize2Icon} boxSize="3.5" />
            </IconButton>
          </Tooltip>
          <Tooltip content={t('widgets.floating.dock')}>
            <IconButton
              aria-label={t('widgets.floating.dock')}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={handleDock}
            >
              <Icon as={PanelRightIcon} boxSize="3.5" />
            </IconButton>
          </Tooltip>
        </HStack>
      </HStack>
      {isShaded ? null : (
        <Flex direction="column" flex="1" minH="0" overflow="hidden">
          {isEnabled && widget ? (
            <WidgetRendererById instanceId={instance.id} region="floating" widget={widget} />
          ) : (
            <HStack color="fg.error" gap="1.5" p="3">
              <Icon as={TriangleAlertIcon} boxSize="3.5" />
              <Text fontSize="xs">{t('widgets.failure.title', { label })}</Text>
            </HStack>
          )}
        </Flex>
      )}
      {isShaded || isMaximized ? null : (
        <Box
          aria-label={t('widgets.floating.resize')}
          aria-valuemin={FLOATING_MIN_WIDTH_PX}
          aria-valuenow={geometry.widthPx}
          bottom="0"
          cursor="nwse-resize"
          h="4"
          position="absolute"
          right="0"
          role="separator"
          tabIndex={0}
          touchAction="none"
          w="4"
          onKeyDown={handleResizeKeyDown}
          onPointerDown={handleResizePointerDown}
        />
      )}
    </Flex>
  );
};
