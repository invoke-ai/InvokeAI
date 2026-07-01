/* eslint-disable react/react-compiler */
import type {
  WidgetInstanceId,
  WidgetInstanceRuntimeMeta,
  WidgetManifest,
  WidgetRegion,
  WidgetRuntimeApi,
  WidgetTypeId,
  WorkbenchRegion,
} from '@workbench/types';

import { Box, Flex, HStack, Icon, Stack, Text, type RecipeVariantProps, useRecipe } from '@chakra-ui/react';
import { chipRecipe } from '@theme/recipes';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { SettingsIcon, type LucideIcon } from 'lucide-react';
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

const PANEL_SIZE_STEP_PX = 16;
const MIN_PANEL_SIZE_PX = 180;
const MAX_PANEL_SIZE_PX = 520;
const MIN_BOTTOM_PANEL_SIZE_PX = 96;
const MAX_BOTTOM_PANEL_SIZE_PX = 420;
const RESIZE_HANDLE_HOVER_PROPS = { bg: 'accent.solid', opacity: 0.45 };
const RESIZE_HANDLE_FOCUS_PROPS = { bg: 'accent.solid', opacity: 0.65, outline: '2px solid {colors.accent.solid}' };

const getPanelSizeBounds = (region: WidgetRegion): { max: number; min: number } => {
  if (region === 'bottom') {
    return { max: MAX_BOTTOM_PANEL_SIZE_PX, min: MIN_BOTTOM_PANEL_SIZE_PX };
  }

  return { max: MAX_PANEL_SIZE_PX, min: MIN_PANEL_SIZE_PX };
};

const clampSize = (region: WidgetRegion, sizePx: number): number => {
  const { max, min } = getPanelSizeBounds(region);

  return Math.min(max, Math.max(min, sizePx));
};

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
  const regionState = useActiveProjectSelector((project) => project.widgetRegions[region]);
  const dispatch = useWorkbenchDispatch();
  const [dragSizePx, setDragSizePx] = useState<number | null>(null);
  const isLeft = region === 'left';
  const isBottom = region === 'bottom';
  const displaySizePx = dragSizePx ?? regionState.sizePx;
  const sizeBounds = getPanelSizeBounds(region);
  const focusRegionProps = useFocusRegionProps(region);

  const commitSize = useCallback(
    (sizePx: number) => {
      const nextSizePx = clampSize(region, sizePx);

      if (nextSizePx !== regionState.sizePx) {
        dispatch({ region, sizePx: nextSizePx, type: 'setRegionWidgetSize' });
      }
    },
    [dispatch, region, regionState.sizePx]
  );

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();

      const startX = event.clientX;
      const startY = event.clientY;
      const startSizePx = regionState.sizePx;
      let nextSizePx = startSizePx;
      const direction = isLeft ? 1 : -1;

      const handlePointerMove = (moveEvent: PointerEvent) => {
        const deltaPx = isBottom ? startY - moveEvent.clientY : (moveEvent.clientX - startX) * direction;

        nextSizePx = clampSize(region, startSizePx + deltaPx);
        setDragSizePx(nextSizePx);
      };

      const handlePointerUp = () => {
        window.removeEventListener('pointermove', handlePointerMove);
        window.removeEventListener('pointerup', handlePointerUp);
        window.removeEventListener('pointercancel', handlePointerUp);
        setDragSizePx(null);
        commitSize(nextSizePx);
      };

      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
      window.addEventListener('pointercancel', handlePointerUp);
    },
    [commitSize, isBottom, isLeft, region, regionState.sizePx]
  );

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const step = event.shiftKey ? PANEL_SIZE_STEP_PX * 2 : PANEL_SIZE_STEP_PX;
      const sizeChanges: Partial<Record<string, number>> = isBottom
        ? { ArrowDown: -step, ArrowUp: step, End: sizeBounds.max - displaySizePx, Home: sizeBounds.min - displaySizePx }
        : {
            ArrowLeft: isLeft ? -step : step,
            ArrowRight: isLeft ? step : -step,
            End: sizeBounds.max - displaySizePx,
            Home: sizeBounds.min - displaySizePx,
          };
      const sizeChange = sizeChanges[event.key];

      if (sizeChange === undefined) {
        return;
      }

      event.preventDefault();
      commitSize(displaySizePx + sizeChange);
    },
    [commitSize, displaySizePx, isBottom, isLeft, sizeBounds.max, sizeBounds.min]
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
        aria-valuemax={sizeBounds.max}
        aria-valuemin={sizeBounds.min}
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

export const WidgetHeader = ({
  actions,
  instance,
  manifest,
  region,
  runtime,
}: {
  actions?: ReactNode;
  instance: WidgetInstanceRuntimeMeta;
  manifest: WidgetManifest;
  region: WorkbenchRegion;
  runtime: WidgetRuntimeApi;
}) => {
  const { t } = useTranslation();
  // Manifests may provide a component label (e.g. Workflow's editable
  // `Workflow / [name]`); plain strings render as the standard title.
  const HeaderLabel = manifest.headerLabel;
  const label = resolveWidgetLabel(manifest, t);
  const handleSettingsClick = useCallback(
    () => openWorkbenchSettings(manifest.settingsSection),
    [manifest.settingsSection]
  );

  return (
    <HStack justify="space-between" borderBottomWidth={1} h={10} ps="3" pe="2">
      <HStack flex="1" gap="1.5" minW="0">
        {HeaderLabel ? (
          <HeaderLabel region={region} />
        ) : (
          <Text fontSize="xs" fontWeight="700">
            {label}
          </Text>
        )}
      </HStack>
      <HStack flexShrink={0} gap="1.5">
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
        <WidgetActionsMenu instance={instance} manifest={manifest} region={region} runtime={runtime} />
      </HStack>
    </HStack>
  );
};

export const FieldPlaceholder = ({ label, h }: { label: string; h: string }) => (
  <Stack gap="1">
    <Text color="fg.muted" fontSize="2xs" fontWeight="600" textTransform="uppercase">
      {label}
    </Text>
    <Box bg="bg.subtle" borderWidth="1px" borderColor="border.subtle" h={h} rounded="md" w="full" />
  </Stack>
);

export const StatusWidgetChip = ({
  children,
  icon,
  tone,
}: {
  children: ReactNode;
  icon: LucideIcon;
  tone?: NonNullable<RecipeVariantProps<typeof chipRecipe>>['tone'];
}) => {
  const recipe = useRecipe({ recipe: chipRecipe });

  return (
    <HStack css={recipe({ tone })}>
      <Icon as={icon} boxSize="3" />
      <Text whiteSpace="nowrap">{children}</Text>
    </HStack>
  );
};
