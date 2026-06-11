import { Badge, Box, Flex, Group, Icon, Menu, NumberInput, Portal } from '@chakra-ui/react';
import { ChevronDownIcon, ListOrderedIcon, PauseIcon, PlayIcon, XIcon } from 'lucide-react';

import { InvokeControl } from './InvokeControl';
import { LayoutPresetMenu } from './LayoutPresetMenu';
import { ProjectTabs } from './ProjectTabs';
import { IconButton } from './ui/Button';
import { AccountMenu } from '../auth/components/AccountMenu';
import { useWorkbenchPreferences } from '../settings/store';
import { useWorkbench } from '../WorkbenchContext';
import { DEFAULT_THEME_ID, THEMES_BY_ID } from '../../theme/themes';
import { Tooltip } from './ui/Tooltip';
import { Link } from '@tanstack/react-router';

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
    <QueueInfo />
    <QueueActions />
    <Box w="1px" h="5" bg="border.subtle" mx="1" flexShrink={0} />
    <ProjectTabs />
    <LayoutPresetMenu />
    <AccountMenu />
  </Flex>
);

/** Compact legacy Invoke logo used as the workbench app-menu affordance. */
const BrandMark = () => {
  const { themeId } = useWorkbenchPreferences();
  const theme = THEMES_BY_ID[themeId] ?? THEMES_BY_ID[DEFAULT_THEME_ID];

  return (
    <Link
      to="/"
      style={{
        flexShrink: 0,
        height: '100%',
        aspectRatio: '1/1',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <svg aria-hidden="true" fill="none" height="20" viewBox="0 0 44 44" width="20">
        <path
          d="M29.1951 10.6667H42V2H2V10.6667H14.8049L29.1951 33.3333H42V42H2V33.3333H14.8049"
          stroke={theme.colors.accent}
          strokeWidth="2.8"
        />
      </svg>
    </Link>
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

const QueueInfo = () => {
  const { activeProject } = useWorkbench();
  const activeCount = activeProject.queue.items.filter(
    (item) => item.status === 'pending' || item.status === 'running'
  ).length;
  const totalCount = activeProject.queue.items.length;

  return (
    <Badge variant="outline" h="9" borderColor="border.subtle" fontSize="xs" fontWeight="700" gap="1" px="2">
      <ListOrderedIcon size="14" />
      {activeCount}/{totalCount}
    </Badge>
  );
};

/**
 * Queue status + cancel cluster placeholder.
 *
 * Mirrors the spec's queue progress / cancel affordance. Real queue wiring,
 * snapshotting, and cancellation arrive with the Invocation Controller phases.
 */
const QueueActions = () => {
  const queueEndActions = [
    {
      label: 'Cancel Current Item',
      icon: XIcon,
      onClick: () => void 0,
    },
    {
      label: 'Cancel All Items',
      icon: XIcon,
      onClick: () => void 0,
    },
    {
      label: 'Cancel all except current item',
      icon: XIcon,
      onClick: () => void 0,
    },
  ];

  const queueProcessorActions = [
    {
      label: 'Resume Processor',
      icon: PlayIcon,
      onClick: () => void 0,
    },
    {
      label: 'Pause Processor',
      icon: PauseIcon,
      onClick: () => void 0,
    },
    {
      label: 'Open Queue',
      icon: ListOrderedIcon,
      onClick: () => void 0,
    },
  ];

  return (
    <Menu.Root>
      <Group attached>
        <Tooltip content="Cancel current item" showArrow>
          <IconButton variant="outline" size="sm" roundedEnd="none">
            <XIcon />
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
            <ChevronDownIcon />
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
  );
};
