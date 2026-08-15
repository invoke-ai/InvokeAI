import type { WidgetRegion } from '@workbench/layoutContracts';
import type {
  WidgetInstanceId,
  WidgetInstanceRuntimeMeta,
  WidgetHeaderLabel,
  WidgetHeaderMenu,
  WidgetManifest,
  WidgetRuntimeApi,
  WidgetTypeId,
  WorkbenchRegion,
} from '@workbench/widgetContracts';

import { Box, Flex, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { resolveWidgetInstanceLabel } from '@workbench/widgetLabels';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { clampPanelSize, getPanelSizeBounds } from '@workbench/workbenchState';
import { SettingsIcon } from 'lucide-react';
import {
  useCallback,
  useMemo,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';
import { useTranslation } from 'react-i18next';

import { WidgetActionsMenu } from './WidgetActionsMenu';
import { WidgetIdentityIcon } from './WidgetIdentityIcon';
import { WidgetSourceLockBadge } from './WidgetSourceLockBadge';

const PANEL_SIZE_STEP_PX = 16;
const RESIZE_HANDLE_HOVER_PROPS = { bg: 'accent.solid', opacity: 0.45 };
const RESIZE_HANDLE_FOCUS_PROPS = { bg: 'accent.solid', opacity: 0.65, outline: '2px solid {colors.accent.solid}' };

export const WidgetPanelFrame = ({
  children,
  instanceId,
  region,
  typeId,
}: {
  children: ReactNode;
  instanceId?: WidgetInstanceId;
  region: Exclude<WidgetRegion, 'center'>;
  typeId?: WidgetTypeId;
}) => {
  const { t } = useTranslation();
  const regionState = useActiveProjectSelector((project) => project.widgetRegions[region]);
  const { layout } = useWorkbenchCommands();
  const [dragSizePx, setDragSizePx] = useState<number | null>(null);
  const isLeft = region === 'left';
  const isBottom = region === 'bottom';
  // Clamped at render, not just on commit, so a persisted size from before a
  // bounds change heals on screen immediately instead of on the next resize.
  const displaySizePx = clampPanelSize(region, dragSizePx ?? regionState.sizePx);
  const { max: maxPanelSizePx, min: minPanelSizePx } = getPanelSizeBounds(region);
  const focusRegionProps = useFocusRegionProps(region);

  const commitSize = useCallback(
    (sizePx: number) => {
      const nextSizePx = clampPanelSize(region, sizePx);

      if (nextSizePx !== regionState.sizePx) {
        layout.setRegionSize(region, nextSizePx);
      }
    },
    [layout, region, regionState.sizePx]
  );

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();

      const startX = event.clientX;
      const startY = event.clientY;
      const startSizePx = regionState.sizePx;
      let nextSizePx = startSizePx;
      const direction = isLeft ? 1 : -1;
      const pointerSession = new AbortController();

      const handlePointerMove = (moveEvent: PointerEvent) => {
        const deltaPx = isBottom ? startY - moveEvent.clientY : (moveEvent.clientX - startX) * direction;

        nextSizePx = clampPanelSize(region, startSizePx + deltaPx);
        setDragSizePx(nextSizePx);
      };

      const handlePointerUp = () => {
        pointerSession.abort();
        setDragSizePx(null);
        commitSize(nextSizePx);
      };

      window.addEventListener('pointermove', handlePointerMove, { signal: pointerSession.signal });
      window.addEventListener('pointerup', handlePointerUp, { signal: pointerSession.signal });
      window.addEventListener('pointercancel', handlePointerUp, { signal: pointerSession.signal });
    },
    [commitSize, isBottom, isLeft, region, regionState.sizePx]
  );

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const step = event.shiftKey ? PANEL_SIZE_STEP_PX * 2 : PANEL_SIZE_STEP_PX;
      const sizeChanges: Partial<Record<string, number>> = isBottom
        ? { ArrowDown: -step, ArrowUp: step, End: maxPanelSizePx - displaySizePx, Home: minPanelSizePx - displaySizePx }
        : {
            ArrowLeft: isLeft ? -step : step,
            ArrowRight: isLeft ? step : -step,
            End: maxPanelSizePx - displaySizePx,
            Home: minPanelSizePx - displaySizePx,
          };
      const sizeChange = sizeChanges[event.key];

      if (sizeChange === undefined) {
        return;
      }

      event.preventDefault();
      commitSize(displaySizePx + sizeChange);
    },
    [commitSize, displaySizePx, isBottom, isLeft, maxPanelSizePx, minPanelSizePx]
  );
  const panelSizeProps = useMemo(
    () => (isBottom ? { h: `${displaySizePx}px`, w: 'full' } : { h: 'full', w: `${displaySizePx}px` }),
    [displaySizePx, isBottom]
  );
  const resizeOrientationProps = useMemo(
    () => (isBottom ? { h: '2', left: '0', right: '0', top: '-1' } : { bottom: '0', top: '0', w: '2' }),
    [isBottom]
  );
  const resizeSideProps = useMemo(
    () => (!isBottom ? (isLeft ? { right: '-1' } : { left: '-1' }) : {}),
    [isBottom, isLeft]
  );

  return (
    <Flex
      aria-label={t('widgets.panelLabel', { region })}
      as="aside"
      bg="bg.subtle"
      borderColor="border.subtle"
      borderRightWidth={isLeft ? '1px' : '0'}
      borderLeftWidth={!isLeft && !isBottom ? '1px' : '0'}
      borderTopWidth={isBottom ? '1px' : '0'}
      direction="column"
      flexShrink={0}
      overflow="hidden"
      minW="0"
      data-hotkey-widget-instance-id={instanceId}
      data-hotkey-widget-region={region}
      data-hotkey-widget-type-id={typeId}
      {...focusRegionProps}
      {...panelSizeProps}
    >
      {children}
      <Box
        aria-label={`Resize ${region} widget panel`}
        aria-orientation={isBottom ? 'horizontal' : 'vertical'}
        aria-valuemax={maxPanelSizePx}
        aria-valuemin={minPanelSizePx}
        aria-valuenow={displaySizePx}
        as="div"
        cursor={isBottom ? 'ns-resize' : 'ew-resize'}
        opacity="0"
        position="absolute"
        role="separator"
        tabIndex={0}
        transition="opacity var(--wb-motion-duration-fast) ease, background var(--wb-motion-duration-fast) ease"
        zIndex="1"
        {...resizeOrientationProps}
        {...resizeSideProps}
        _hover={RESIZE_HANDLE_HOVER_PROPS}
        _focusVisible={RESIZE_HANDLE_FOCUS_PROPS}
        onKeyDown={handleKeyDown}
        onPointerDown={handlePointerDown}
      />
    </Flex>
  );
};

/**
 * The trailing action cluster of a widget's chrome: the manifest's own
 * `headerActions`, the settings gear, and the shared overflow menu. Panels
 * render it inside {@link WidgetHeader}; the center region renders it on its
 * own, floating over the work surface, so it lives apart from the header row.
 */
export const WidgetHeaderActionsGroup = ({
  actions,
  HeaderMenu,
  instance,
  manifest,
  region,
  runtime,
}: {
  actions?: ReactNode;
  HeaderMenu?: WidgetHeaderMenu;
  instance: WidgetInstanceRuntimeMeta;
  manifest: WidgetManifest;
  region: WorkbenchRegion;
  runtime: WidgetRuntimeApi;
}) => {
  const { t } = useTranslation();
  const label = resolveWidgetInstanceLabel(instance, manifest, t);
  const handleSettingsClick = useCallback(
    () => openWorkbenchSettings(manifest.settingsSection),
    [manifest.settingsSection]
  );

  return (
    <HStack flexShrink={0} gap="0.5">
      {actions}
      {manifest.settingsSection ? (
        <Tooltip content={t('widgets.settingsLabel', { label })}>
          <IconButton
            aria-label={t('widgets.settingsLabel', { label })}
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={handleSettingsClick}
          >
            <Icon as={SettingsIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      ) : null}
      <WidgetActionsMenu
        HeaderMenu={HeaderMenu}
        instance={instance}
        manifest={manifest}
        region={region}
        runtime={runtime}
      />
    </HStack>
  );
};

export const WidgetHeader = ({
  actions,
  HeaderLabel,
  HeaderMenu,
  instance,
  manifest,
  region,
  runtime,
}: {
  actions?: ReactNode;
  HeaderLabel?: WidgetHeaderLabel;
  HeaderMenu?: WidgetHeaderMenu;
  instance: WidgetInstanceRuntimeMeta;
  manifest: WidgetManifest;
  region: WorkbenchRegion;
  runtime: WidgetRuntimeApi;
}) => {
  const { t } = useTranslation();
  // Manifests may provide a component label (e.g. Workflow's editable
  // `Workflow / [name]`); plain strings render as the standard title.
  const label = resolveWidgetInstanceLabel(instance, manifest, t);

  return (
    <HStack justify="space-between" borderBottomWidth={1} h={10} ps="3" pe="2">
      <HStack flex="1" gap="1.5" minW="0">
        <WidgetIdentityIcon icon={manifest.icon} />
        {HeaderLabel && !instance.title ? (
          <HeaderLabel region={region} />
        ) : (
          <Text data-widget-identity-label="" fontSize="xs" fontWeight="700">
            {label}
          </Text>
        )}
        <WidgetSourceLockBadge typeId={manifest.id} />
      </HStack>
      <WidgetHeaderActionsGroup
        HeaderMenu={HeaderMenu}
        actions={actions}
        instance={instance}
        manifest={manifest}
        region={region}
        runtime={runtime}
      />
    </HStack>
  );
};

export const WidgetTooltipFrame = ({
  children,
  icon,
  isLoading = false,
}: {
  children: ReactNode;
  icon: WidgetManifest['icon'];
  isLoading?: boolean;
}) => (
  <HStack align="start" gap="1.5" minW="9rem">
    <WidgetIdentityIcon icon={icon} isLoading={isLoading} />
    <Box minW="0">{children}</Box>
  </HStack>
);

export const FieldPlaceholder = ({ label, h }: { label: string; h: string }) => (
  <Stack gap="1">
    <Text color="fg.muted" fontSize="2xs" fontWeight="600" textTransform="uppercase">
      {label}
    </Text>
    <Box bg="bg.subtle" borderWidth="1px" borderColor="border.subtle" h={h} rounded="md" w="full" />
  </Stack>
);
