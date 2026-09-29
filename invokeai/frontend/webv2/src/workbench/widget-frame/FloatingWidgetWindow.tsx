import type { FloatingWidgetState } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { Flex, HStack, Icon, Separator, Text } from '@chakra-ui/react';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { IconButton } from '@platform/ui/Button';
import { type PointerDragEnd, ResizeCorner, usePointerDrag } from '@platform/ui/ResizeHandle';
import { Tooltip } from '@platform/ui/Tooltip';
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
  useMemo,
  useRef,
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

// CSS clamp keeps a grabbable sliver on-screen even for geometry persisted on a larger display (or after the
// browser window shrinks); the commit clamp only covers drags on the current viewport.
const toWindowPosition = (geometry: FloatingGeometry) => ({
  height: `${geometry.heightPx}px`,
  left: `clamp(${48 - geometry.widthPx}px, ${geometry.x}px, calc(100vw - 48px))`,
  top: `clamp(0px, ${geometry.y}px, calc(100vh - 48px))`,
  width: `${geometry.widthPx}px`,
});

/**
 * Isolate arbitrary widget controls from title-bar drag and double-click shade gestures; not every control is a
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
  const windowRef = useRef<HTMLDivElement>(null);
  const liveGeometryRef = useRef<FloatingGeometry | null>(null);
  const startDrag = usePointerDrag();

  const widget = instance ? getWidgetById(instance.typeId) : undefined;

  const commitGeometry = useCallback(
    (geometry: FloatingGeometry) => {
      const clamped = clampWindowToViewport(geometry, { height: window.innerHeight, width: window.innerWidth });
      widgets.setFloatingGeometry(instanceId, clamped);
    },
    [instanceId, widgets]
  );

  // A gesture writes geometry inline so it renders without React; the committed render then replaces it.
  const writeLiveGeometry = useCallback(
    (geometry: FloatingGeometry | null) => {
      const element = windowRef.current;
      liveGeometryRef.current = geometry;
      if (!element) {
        return;
      }
      const position = geometry ? toWindowPosition(geometry) : null;
      for (const property of ['left', 'top', 'width', 'height'] as const) {
        if (position && !(property === 'height' && state.mode === 'shaded')) {
          element.style.setProperty(property, position[property]);
        } else {
          element.style.removeProperty(property);
        }
      }
    },
    [state.mode]
  );
  const endGesture = useCallback(
    (reason: PointerDragEnd) => {
      const geometry = liveGeometryRef.current;
      if (reason === 'escape') {
        writeLiveGeometry(null);
        return;
      }
      // Scheduled first so a throwing commit still clears the preview.
      requestAnimationFrame(() => writeLiveGeometry(null));
      if (geometry) {
        commitGeometry(geometry);
      }
    },
    [commitGeometry, writeLiveGeometry]
  );
  const cancelGesture = useCallback(() => writeLiveGeometry(null), [writeLiveGeometry]);
  const startGeometry: FloatingGeometry = useMemo(
    () => ({ heightPx: state.heightPx, widthPx: state.widthPx, x: state.x, y: state.y }),
    [state.heightPx, state.widthPx, state.x, state.y]
  );

  const handleTitlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0 || state.mode === 'maximized' || (event.target as HTMLElement).closest('button')) {
        return;
      }

      startDrag(event, {
        cursor: 'move',
        onEnd: endGesture,
        onMove: (deltaX, deltaY) =>
          writeLiveGeometry({ ...startGeometry, x: startGeometry.x + deltaX, y: startGeometry.y + deltaY }),
      });
    },
    [endGesture, startDrag, startGeometry, state.mode, writeLiveGeometry]
  );

  const handleResizeMove = useCallback(
    (deltaX: number, deltaY: number) =>
      writeLiveGeometry({
        ...startGeometry,
        heightPx: Math.max(FLOATING_MIN_HEIGHT_PX, startGeometry.heightPx + deltaY),
        widthPx: Math.max(FLOATING_MIN_WIDTH_PX, startGeometry.widthPx + deltaX),
      }),
    [startGeometry, writeLiveGeometry]
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
  // Flush drafts before docking remounts the widget; registry cleanup only removes flushers.
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

  // Retain window chrome for missing or failed widgets so users can dock them back to the retry surface.
  const isEnabled = widget?.status === 'enabled';
  const label = widget ? resolveWidgetInstanceLabel(instance, widget.manifest, t) : (instance.title ?? instance.id);
  const position = toWindowPosition(state);
  const isMaximized = state.mode === 'maximized';
  const isShaded = state.mode === 'shaded';
  const positionProps = isMaximized
    ? { h: '100vh', left: 0, top: 0, w: '100vw' }
    : { h: isShaded ? 'auto' : position.height, left: position.left, top: position.top, w: position.width };

  return (
    <Flex
      ref={windowRef}
      bg="bg.subtle"
      borderColor="border.emphasized"
      borderWidth="1px"
      direction="column"
      overflow="hidden"
      position="fixed"
      rounded={isMaximized ? 'none' : 'md'}
      shadow="xl"
      zIndex={FLOATING_BASE_Z_INDEX + stackRank}
      // Mark floating widget identity and region so hotkeys target it rather than the last focused docked widget.
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
          {/*
           * Render widget actions and settings in a row because floating content has no frame header; window
           * controls already own layout actions.
           */}
          {isEnabled && widget ? (
            <FloatingChromeBoundary>
              <Suspense fallback={null}>
                <HStack gap="1" onDoubleClick={stopChromeEvent} onPointerDown={stopChromeEvent}>
                  <WidgetChromeSlotById instanceId={instanceId} region="floating" slot="viewActions" widget={widget} />
                </HStack>
              </Suspense>
            </FloatingChromeBoundary>
          ) : null}
          {isEnabled && widget ? <Separator h="4" mx="0.5" orientation="vertical" /> : null}
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
        <ResizeCorner
          label={t('widgets.floating.resize')}
          valueMin={FLOATING_MIN_WIDTH_PX}
          valueNow={state.widthPx}
          onDragCancel={cancelGesture}
          onDragEnd={endGesture}
          onDragMove={handleResizeMove}
          onKeyDown={handleResizeKeyDown}
        />
      )}
    </Flex>
  );
};
