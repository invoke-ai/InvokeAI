import type { GraphWidgetSource } from '@workbench/graphWidgets';
import type { InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';

import { Box, Icon, Menu, Portal, Stack, Status, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { describeRoute, getWidgetTypeIdForSourceId } from '@workbench/graphWidgets';
import { WidgetIcon } from '@workbench/iconResolver';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { LockKeyholeIcon, UnlockKeyholeIcon } from 'lucide-react';
import { useCallback, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationState } from './useInvocationState';

import { RoutingDestinationSegments } from './RoutingDestinationSegments';

const MENU_POSITIONING = { placement: 'bottom-end' } as const;

export const RoutingControl = ({ state }: { state: InvocationState }) => {
  const { t } = useTranslation();
  const { generation } = useWorkbenchCommands();
  const triggerId = useId();
  const triggerIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  const { invocation, placedTypeIds, sources } = state;
  const isLocked = invocation.sourceLocked || invocation.destinationLocked;
  const sourceTypeId = getWidgetTypeIdForSourceId(invocation.sourceId);
  const hasSource = placedTypeIds.has(sourceTypeId);
  const isSameRoute = hasSource && sourceTypeId === invocation.destination;
  const accessibleName = describeRoute({
    destination: invocation.destination,
    destinationLocked: invocation.destinationLocked,
    hasSource,
    sourceId: invocation.sourceId,
    sourceLocked: invocation.sourceLocked,
  });
  const lockActionLabel = t(isLocked ? 'topbar.routing.unlockRouting' : 'topbar.routing.lockRouting');

  const getWidgetIconProps = (type: 'source' | 'destination') => ({
    boxSize: '3.5',
    color: 'fg.muted',
    [`data-routing-${type}-icon`]: '',
    icon: getWidgetById(type === 'source' ? sourceTypeId : invocation.destination)?.manifest.icon,
    position: 'absolute',
    top: type === 'source' ? '4px' : undefined,
    left: type === 'source' ? '4px' : undefined,
    bottom: type === 'destination' ? '4px' : undefined,
    right: type === 'destination' ? '4px' : undefined,
  });

  return (
    <Menu.Root ids={triggerIds} positioning={MENU_POSITIONING}>
      <Tooltip content={t('topbar.routing.change')} ids={triggerIds}>
        <Menu.Trigger asChild>
          <Button
            aria-label={accessibleName}
            data-routing-control=""
            flexShrink={0}
            maxW="36px"
            minW="36px"
            overflow="visible"
            p="0"
            position="relative"
            size="xs"
            variant="outline"
            w="34px"
            display="grid"
          >
            {isSameRoute ? (
              <WidgetIcon
                boxSize="3.5"
                color="fg.muted"
                data-routing-shared-icon=""
                icon={getWidgetById(sourceTypeId)?.manifest.icon}
                placeSelf="center"
              />
            ) : (
              <>
                {hasSource ? (
                  <WidgetIcon {...getWidgetIconProps('source')} />
                ) : (
                  <Box
                    borderColor="border.emphasized"
                    borderStyle="dashed"
                    borderWidth="1px"
                    boxSize="3.5"
                    data-routing-source-icon=""
                    left="4px"
                    position="absolute"
                    rounded="xs"
                    top="4px"
                  />
                )}
                <WidgetIcon {...getWidgetIconProps('destination')} />
              </>
            )}
            {isLocked ? (
              <Status.Root
                colorPalette="accent"
                data-routing-lock-indicator=""
                pointerEvents="none"
                position="absolute"
                right="-1px"
                size="sm"
                top="-1px"
                zIndex="1"
              >
                <Status.Indicator />
              </Status.Root>
            ) : null}
          </Button>
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="17rem">
            <RoutingSectionHeader label={t('topbar.routing.source')} />
            <SourceRadioGroup sourceId={invocation.sourceId} sources={sources} />
            <Menu.Separator />
            <RoutingSectionHeader label={t('topbar.routing.destination')} />
            <Stack px="3" pb="2">
              <DestinationSegments destination={invocation.destination} />
            </Stack>
            <Menu.Separator />
            <Menu.Item value={isLocked ? 'unlock-routing' : 'lock-routing'} onClick={generation.toggleRoutingLock}>
              <Icon as={isLocked ? UnlockKeyholeIcon : LockKeyholeIcon} boxSize="3.5" color="fg.subtle" />
              <Menu.ItemText>{lockActionLabel}</Menu.ItemText>
            </Menu.Item>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const RoutingSectionHeader = ({ label }: { label: string }) => (
  <Text color="fg.subtle" fontSize="2xs" fontWeight="700" px="3" pt="2" pb="1" textTransform="uppercase">
    {label}
  </Text>
);

const SourceRadioGroup = ({
  sourceId,
  sources,
}: {
  sourceId: InvocationSourceId;
  sources: readonly GraphWidgetSource[];
}) => {
  const { generation } = useWorkbenchCommands();
  const handleChange = useCallback(
    (event: { value: string }) => generation.setSource(event.value as InvocationSourceId),
    [generation]
  );

  return (
    <Menu.RadioItemGroup value={sourceId} onValueChange={handleChange}>
      {sources.map((source) => (
        <SourceRow key={source.sourceId} source={source} />
      ))}
    </Menu.RadioItemGroup>
  );
};

const SourceRow = ({ source }: { source: GraphWidgetSource }) => (
  <Menu.RadioItem value={source.sourceId}>
    <Menu.ItemIndicator />
    <WidgetIcon boxSize="3.5" icon={getWidgetById(source.typeId)?.manifest.icon} />
    <Menu.ItemText flex="1">{source.label}</Menu.ItemText>
  </Menu.RadioItem>
);

const DestinationSegments = ({ destination }: { destination: ResultDestination }) => {
  const { generation } = useWorkbenchCommands();
  const { t } = useTranslation();
  const handleChange = useCallback(
    (nextDestination: ResultDestination) => generation.setDestination(nextDestination),
    [generation]
  );

  return (
    <RoutingDestinationSegments
      ariaLabel={t('topbar.routing.destination')}
      value={destination}
      onChange={handleChange}
    />
  );
};
