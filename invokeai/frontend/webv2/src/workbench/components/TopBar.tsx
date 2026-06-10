import { Badge, Box, Flex, Group, HStack, Icon, Menu, NumberInput, Portal, Text } from '@chakra-ui/react';
import {
  PiCaretDownBold,
  PiCubeBold,
  PiListNumbersBold,
  PiPause,
  PiPlay,
  PiUserCircleBold,
  PiXBold,
} from 'react-icons/pi';

import { InvokeControl } from './InvokeControl';
import { LayoutPresetMenu } from './LayoutPresetMenu';
import { ProjectTabs } from './ProjectTabs';
import { SettingsButton } from './SettingsDialog';
import { IconButton } from './ui/Button';
import { useWorkbench } from '../WorkbenchContext';
import { DEFAULT_THEME_ID, THEMES_BY_ID } from '../../theme/themes';
import { Tooltip } from './ui/Tooltip';

/** Workbench top bar: brand, global Invoke command cluster, project tabs, layout + account controls. */
export const TopBar = () => (
  <Flex
    align="center"
    as="header"
    bg="bg.surfaceRaised"
    borderBottomWidth="1px"
    borderColor="border.subtle"
    flexShrink={0}
    gap="2"
    h="12"
    pe="3"
    w="full"
  >
    <BrandMark />
    <InvokeControl />
    <BatchCountField />
    <QueueCluster />
    <Box w="1px" h="5" bg="border.subtle" mx="1" flexShrink={0} />
    <ProjectTabs />
    <LayoutPresetMenu />
    <HStack gap="0.5" flexShrink={0}>
      <IconButton
        aria-label="Model manager"
        color="fg.muted"
        size="sm"
        variant="ghost"
        _hover={{ color: 'fg.default' }}
      >
        <PiCubeBold />
      </IconButton>
      <SettingsButton />
    </HStack>
    <HStack as="button" gap="1.5" color="fg.default" flexShrink={0} px="1">
      <Icon as={PiUserCircleBold} boxSize="4" />
      <Text fontSize="xs" fontWeight="600">
        Josh
      </Text>
    </HStack>
  </Flex>
);

/** Compact legacy Invoke logo used as the workbench app-menu affordance. */
const BrandMark = () => {
  const { state } = useWorkbench();
  const theme = THEMES_BY_ID[state.account.preferences.themeId] ?? THEMES_BY_ID[DEFAULT_THEME_ID];

  return (
    <Flex
      align="center"
      aria-label="Invoke"
      as="button"
      flexShrink={0}
      h="full"
      justify="center"
      rounded="md"
      aspectRatio="1/1"
      me="-1"
    >
      <svg aria-hidden="true" fill="none" height="20" viewBox="0 0 44 44" width="20">
        <path
          d="M29.1951 10.6667H42V2H2V10.6667H14.8049L29.1951 33.3333H42V42H2V33.3333H14.8049"
          stroke={theme.colors.accent}
          strokeWidth="2.8"
        />
      </svg>
    </Flex>
  );
};

const getBatchCount = (values: Record<string, unknown>): number => {
  const batchCount = values.batchCount;

  return typeof batchCount === 'number' && Number.isFinite(batchCount) ? batchCount : 1;
};

const BatchCountField = () => {
  const { activeProject, dispatch } = useWorkbench();
  const batchCount = getBatchCount(activeProject.widgetStates.generate.values);

  return (
    <NumberInput.Root
      allowMouseWheel
      flexShrink={0}
      max={64}
      min={1}
      size="sm"
      value={String(batchCount)}
      w="14"
      onValueChange={({ valueAsNumber }) => {
        if (Number.isFinite(valueAsNumber)) {
          dispatch({ batchCount: valueAsNumber, type: 'setGenerateBatchCount' });
        }
      }}
    >
      <NumberInput.Control />
      <NumberInput.Input paddingStart="4" aria-label="Batch count" />
    </NumberInput.Root>
  );
};

/**
 * Queue status + cancel cluster placeholder.
 *
 * Mirrors the spec's queue progress / cancel affordance. Real queue wiring,
 * snapshotting, and cancellation arrive with the Invocation Controller phases.
 */
const QueueCluster = () => {
  const { activeProject } = useWorkbench();
  const activeCount = activeProject.queue.items.filter(
    (item) => item.status === 'pending' || item.status === 'running'
  ).length;
  const totalCount = activeProject.queue.items.length;

  const queueEndActions = [
    {
      label: 'Cancel Current Item',
      icon: PiXBold,
      onClick: () => void 0,
    },
    {
      label: 'Cancel All Items',
      icon: PiXBold,
      onClick: () => void 0,
    },
    {
      label: 'Cancel all except current item',
      icon: PiXBold,
      onClick: () => void 0,
    },
  ];

  const queueProcessorActions = [
    {
      label: 'Resume Processor',
      icon: PiPlay,
      onClick: () => void 0,
    },
    {
      label: 'Pause Processor',
      icon: PiPause,
      onClick: () => void 0,
    },
    {
      label: 'Open Queue',
      icon: PiListNumbersBold,
      onClick: () => void 0,
    },
  ];

  return (
    <HStack flexShrink={0} gap="1">
      <Badge variant="outline" h="9" borderColor="border.subtle" fontSize="xs" fontWeight="700" gap="1" px="2">
        {/*<Icon as={PiListNumbersBold} boxSize="3" />*/}
        <PiListNumbersBold />
        {activeCount}/{totalCount}
      </Badge>
      <Menu.Root>
        <Group attached>
          <Tooltip content="Cancel current item" showArrow>
            <IconButton variant="outline" size="sm" roundedEnd="none">
              <PiXBold />
            </IconButton>
          </Tooltip>
          <Menu.Trigger>
            <IconButton
              variant="outline"
              size="sm"
              roundedStart="none"
              borderStartWidth="0"
              aspectRatio="unset"
              minW="0"
              w="6"
            >
              <PiCaretDownBold />
            </IconButton>
          </Menu.Trigger>
        </Group>
        <Portal>
          <Menu.Positioner>
            <Menu.Content>
              <Menu.ItemGroup>
                {queueEndActions.map((action, index) => (
                  <Menu.Item
                    key={index}
                    onClick={action.onClick}
                    value={action.label}
                    color="fg.error"
                    _hover={{ bg: 'bg.error', color: 'fg.error' }}
                  >
                    <Icon as={action.icon} boxSize="3" />
                    <span>{action.label}</span>
                  </Menu.Item>
                ))}
              </Menu.ItemGroup>
              <Menu.Separator />
              <Menu.ItemGroup>
                {queueProcessorActions.map((action, index) => (
                  <Menu.Item key={index} onClick={action.onClick} value={action.label}>
                    <Icon as={action.icon} boxSize="3" />
                    <span>{action.label}</span>
                  </Menu.Item>
                ))}
              </Menu.ItemGroup>
            </Menu.Content>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </HStack>
  );
};
