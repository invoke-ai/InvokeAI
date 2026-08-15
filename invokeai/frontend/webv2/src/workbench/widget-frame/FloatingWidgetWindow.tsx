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
  useCallback,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { useTranslation } from 'react-i18next';

import { WidgetRendererById } from './WidgetRenderer';
import { areWidgetRenderInstancesEqual } from './widgetRenderInstance';

/** Below Chakra dialogs/popovers/toasts; above the docked shell. */
const FLOATING_BASE_Z_INDEX = 800;

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

  const beginPointerOperation = useCallback(
    (
      event: ReactPointerEvent<HTMLDivElement>,
      apply: (deltaX: number, deltaY: number, start: FloatingGeometry) => FloatingGeometry
    ) => {
      event.preventDefault();

      const startX = event.clientX;
      const startY = event.clientY;
      const start: FloatingGeometry = { heightPx: state.heightPx, widthPx: state.widthPx, x: state.x, y: state.y };
      let next = start;
      const pointerSession = new AbortController();

      const handlePointerMove = (moveEvent: PointerEvent) => {
        next = apply(moveEvent.clientX - startX, moveEvent.clientY - startY, start);
        setDragGeometry(next);
      };

      const handlePointerUp = () => {
        pointerSession.abort();
        setDragGeometry(null);
        commitGeometry(next);
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

  if (!instance || !widget) {
    return null;
  }

  // A widget that fails registration while floated keeps its chrome. Rendering
  // nothing would strand the instance: it is in no region, so the dock control
  // in this title bar is the only way back to the rail — and to the docked
  // failure card, which owns the retry.
  const isEnabled = widget.status === 'enabled';
  const label = resolveWidgetInstanceLabel(instance, widget.manifest, t);
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
      onPointerDownCapture={handleFocus}
      {...positionProps}
    >
      <HStack
        borderBottomWidth={isShaded ? 0 : '1px'}
        cursor={isMaximized ? 'default' : 'move'}
        flexShrink={0}
        gap="1.5"
        h={10}
        justify="space-between"
        pe="2"
        ps="3"
        // `preventDefault` on pointerdown does not stop touch panning: without
        // this the browser claims the gesture and cancels the drag.
        touchAction="none"
        userSelect="none"
        onDoubleClick={handleTitleDoubleClick}
        onPointerDown={handleTitlePointerDown}
      >
        <HStack flex="1" gap="1.5" minW="0">
          <WidgetIcon boxSize="4" icon={widget.manifest.icon} />
          <Text fontSize="xs" fontWeight="700" truncate>
            {label}
          </Text>
        </HStack>
        <HStack flexShrink={0} gap="1">
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
          {isEnabled ? (
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
          bottom="0"
          cursor="nwse-resize"
          h="4"
          position="absolute"
          right="0"
          touchAction="none"
          w="4"
          onPointerDown={handleResizePointerDown}
        />
      )}
    </Flex>
  );
};
