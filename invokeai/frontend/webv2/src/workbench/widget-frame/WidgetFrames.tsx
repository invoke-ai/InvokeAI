import type { WidgetRegion } from '@workbench/layoutContracts';
import type {
  WidgetInstanceId,
  WidgetInstanceRuntimeMeta,
  WidgetHeaderActions,
  WidgetHeaderLabel,
  WidgetHeaderMenu,
  WidgetManifest,
  WidgetRuntimeApi,
  WidgetTypeId,
  WorkbenchRegion,
} from '@workbench/widgetContracts';

import { Box, Flex, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { IconButton } from '@platform/ui/Button';
import { PanelHeader } from '@platform/ui/PanelHeader';
import { isResizeDragActive, ResizeHandle, subscribeResizeDrag } from '@platform/ui/ResizeHandle';
import { Tooltip } from '@platform/ui/Tooltip';
import { useFocusRegionProps, useHighlightedRegion, useWorkbenchFocus } from '@workbench/focusRegions';
import { isWidgetRegion } from '@workbench/layoutContracts';
import { WidgetSettingsButton } from '@workbench/settings/WidgetSettingsButton';
import { resolveWidgetInstanceLabel } from '@workbench/widgetLabels';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { clampPanelSize, getPanelSizeBounds, getVisiblePanelCollapseThreshold } from '@workbench/workbenchState';
import { PictureInPicture2Icon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { WidgetActionsMenu } from './WidgetActionsMenu';
import { WidgetIdentityIcon } from './WidgetIdentityIcon';
import { WidgetSourceLockBadge } from './WidgetSourceLockBadge';

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
  const isLeft = region === 'left';
  const isBottom = region === 'bottom';
  // Clamped at render, not just on commit, so a persisted size from before a
  // bounds change heals on screen immediately instead of on the next resize.
  const displaySizePx = clampPanelSize(region, regionState.sizePx);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [measuredSizePx, setMeasuredSizePx] = useState<number | null>(null);
  // Side widths are preferences the viewport may squeeze; drags, keyboard floors, and ARIA use the rendered width.
  const bindFrame = useCallback(
    (frame: HTMLDivElement | null) => {
      frameRef.current = frame;
      if (isBottom || !frame || typeof ResizeObserver === 'undefined') {
        return;
      }
      // A drag's live width is not the viewport's squeeze; measure again once it ends.
      const measure = () => {
        if (!isResizeDragActive()) {
          setMeasuredSizePx(Math.round(frame.getBoundingClientRect().width));
        }
      };
      const observer = new ResizeObserver(measure);
      observer.observe(frame);
      const unsubscribeResizeDrag = subscribeResizeDrag(measure);
      return () => {
        unsubscribeResizeDrag();
        observer.disconnect();
      };
    },
    [isBottom]
  );
  const visibleSizePx = measuredSizePx === null ? displaySizePx : Math.min(measuredSizePx, displaySizePx);
  const { max: maxPanelSizePx, min: minPanelSizePx } = getPanelSizeBounds(region);
  const focusRegionProps = useFocusRegionProps(region);
  // This panel's outline, or a side divider's center neighbour, is drawn over the divider. The bottom divider spans
  // the side panels too, so only its own outline hides it.
  const highlightedRegion = useHighlightedRegion();
  const isDividerOutlined = highlightedRegion === region || (highlightedRegion === 'center' && !isBottom);

  const commitSize = useCallback(
    (sizePx: number) => {
      const nextSizePx = clampPanelSize(region, sizePx);

      if (nextSizePx !== regionState.sizePx) {
        layout.setRegionSize(region, nextSizePx);
      }
    },
    [layout, region, regionState.sizePx]
  );
  // Collapse is a visibility change, not a resize: `sizePx` keeps the chosen width so the rail reopens it there.
  const collapse = useMemo(
    () => ({
      // Overshoot is measured from the rendered width when the viewport squeezes the panel below its preference.
      at: getVisiblePanelCollapseThreshold(region, visibleSizePx) + (displaySizePx - visibleSizePx),
      onCollapse: () => layout.setRegionCollapsed(region, true),
      preview: 0,
    }),
    [displaySizePx, layout, region, visibleSizePx]
  );
  const handle = (
    <ResizeHandle
      collapse={collapse}
      label={`Resize ${region} widget panel`}
      lineHidden={isDividerOutlined}
      max={maxPanelSizePx}
      min={minPanelSizePx}
      orientation={isBottom ? 'horizontal' : 'vertical'}
      pane={isLeft ? 'before' : 'after'}
      paneRef={frameRef}
      renderedValue={visibleSizePx}
      value={displaySizePx}
      onCommit={commitSize}
    />
  );

  return (
    // Side widths shrink to protect center space and opposite rails; bottom height stays fixed.
    <Flex
      direction={isBottom ? 'column' : 'row'}
      flexShrink={isBottom ? 0 : 1}
      h={isBottom ? `${displaySizePx}px` : 'full'}
      minW="0"
      ref={bindFrame}
      w={isBottom ? 'full' : `${displaySizePx}px`}
      {...focusRegionProps}
    >
      {isLeft ? null : handle}
      <Flex
        aria-label={t('widgets.panelLabel', { region })}
        as="aside"
        bg="bg.subtle"
        direction="column"
        flex="1"
        minH="0"
        minW="0"
        overflow="hidden"
        data-hotkey-widget-instance-id={instanceId}
        data-hotkey-widget-region={region}
        data-hotkey-widget-type-id={typeId}
      >
        {children}
      </Flex>
      {isLeft ? handle : null}
    </Flex>
  );
};

export const WidgetFloatButton = ({
  instanceId,
  manifest,
  region,
}: {
  instanceId: WidgetInstanceId;
  manifest: WidgetManifest;
  region: WorkbenchRegion;
}) => {
  const { t } = useTranslation();
  const { widgets } = useWorkbenchCommands();
  const { focusFloating } = useWorkbenchFocus();
  const dockableRegion = isWidgetRegion(region) && region !== 'center' ? region : undefined;
  // Flush drafts before floating unmounts the docked view; preserve the clicked region as the multi-region
  // instance's dock origin. Focus follows the widget into its window.
  const handleFloat = useCallback(() => {
    if (!dockableRegion) {
      return;
    }

    flushWorkbenchDrafts();
    widgets.float(instanceId, dockableRegion, { height: window.innerHeight, width: window.innerWidth });
    focusFloating(instanceId);
  }, [dockableRegion, focusFloating, instanceId, widgets]);
  const canFloat = Boolean(manifest.allowFloating) && dockableRegion !== undefined;

  if (!canFloat) {
    return null;
  }

  return (
    <Tooltip content={t('widgets.floating.floatWindow')}>
      <IconButton
        aria-label={t('widgets.floating.floatWindow')}
        color="fg.muted"
        size="sm"
        variant="ghost"
        onClick={handleFloat}
      >
        <Icon as={PictureInPicture2Icon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};

/** Share widget actions, settings, float, and overflow between panel headers and hoisted center chrome. */
export const WidgetHeaderActionsGroup = ({
  actions,
  HeaderMenu,
  SettingsActions,
  instance,
  manifest,
  region,
  runtime,
}: {
  actions?: ReactNode;
  HeaderMenu?: WidgetHeaderMenu;
  SettingsActions?: WidgetHeaderActions;
  instance: WidgetInstanceRuntimeMeta;
  manifest: WidgetManifest;
  region: WorkbenchRegion;
  runtime: WidgetRuntimeApi;
}) => {
  return (
    <HStack flexShrink={0} gap="0.5">
      {actions}
      {manifest.settings ? (
        <WidgetSettingsButton
          SettingsActions={SettingsActions}
          instance={instance}
          manifest={manifest}
          region={region}
          runtime={runtime}
        />
      ) : null}
      <WidgetFloatButton instanceId={instance.id} manifest={manifest} region={region} />
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
  SettingsActions,
  instance,
  manifest,
  region,
  runtime,
}: {
  actions?: ReactNode;
  HeaderLabel?: WidgetHeaderLabel;
  HeaderMenu?: WidgetHeaderMenu;
  SettingsActions?: WidgetHeaderActions;
  instance: WidgetInstanceRuntimeMeta;
  manifest: WidgetManifest;
  region: WorkbenchRegion;
  runtime: WidgetRuntimeApi;
}) => {
  const { t } = useTranslation();
  const label = resolveWidgetInstanceLabel(instance, manifest, t);

  return (
    <PanelHeader>
      <HStack flex="1" gap="1.5" minW="0">
        <WidgetIdentityIcon icon={manifest.icon} />
        {HeaderLabel && !instance.title ? (
          <HeaderLabel region={region} />
        ) : (
          <Text data-widget-identity-label="" fontWeight="700">
            {label}
          </Text>
        )}
        <WidgetSourceLockBadge typeId={manifest.id} />
      </HStack>
      <WidgetHeaderActionsGroup
        HeaderMenu={HeaderMenu}
        SettingsActions={SettingsActions}
        actions={actions}
        instance={instance}
        manifest={manifest}
        region={region}
        runtime={runtime}
      />
    </PanelHeader>
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
    <Text color="fg.muted" fontSize="xs" fontWeight="600" textTransform="uppercase">
      {label}
    </Text>
    <Box bg="bg.subtle" borderWidth="1px" borderColor="border.subtle" h={h} rounded="md" w="full" />
  </Stack>
);
