import { Badge, Box, Flex, HStack, Icon, IconButton, NumberInput, Text } from '@chakra-ui/react';
import { PiCaretDownBold, PiCubeBold, PiListNumbersBold, PiUserCircleBold, PiXBold } from 'react-icons/pi';

import { InvokeControl } from './InvokeControl';
import { LayoutPresetMenu } from './LayoutPresetMenu';
import { ProjectTabs } from './ProjectTabs';
import { SettingsButton } from './SettingsDialog';
import { useWorkbench } from '../WorkbenchContext';

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
    h="10"
    px="3"
    w="full"
  >
    <BrandMark />
    <BatchCountField />
    <InvokeControl />
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

/** Brand mark placeholder for the future workbench logo / app menu. */
const BrandMark = () => (
  <Flex
    align="center"
    aria-label="Invoke"
    as="button"
    color="accent.invoke"
    flexShrink={0}
    h="7"
    justify="center"
    rounded="md"
    w="7"
    _hover={{ bg: 'bg.surface' }}
  >
    <Icon as={PiListNumbersBold} boxSize="5" transform="rotate(90deg)" />
  </Flex>
);

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
      size="xs"
      value={String(batchCount)}
      w="16"
      onValueChange={({ valueAsNumber }) => {
        if (Number.isFinite(valueAsNumber)) {
          dispatch({ batchCount: valueAsNumber, type: 'setGenerateBatchCount' });
        }
      }}
    >
      <NumberInput.Control />
      <NumberInput.Input aria-label="Batch count" />
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

  return (
    <HStack flexShrink={0} gap="1">
      <Badge bg="accent.active" color="accent.activeFg" fontSize="2xs" fontWeight="700" gap="1" h="7" px="2">
        <Icon as={PiListNumbersBold} boxSize="3" />
        {activeCount}/{totalCount}
      </Badge>
      <IconButton
        aria-label="Cancel current batch"
        borderWidth="1px"
        borderColor="border.emphasis"
        color="fg.muted"
        size="xs"
        variant="ghost"
        _hover={{ color: 'fg.default' }}
      >
        <PiXBold />
      </IconButton>
      <IconButton
        aria-label="Queue options"
        borderWidth="1px"
        borderColor="border.emphasis"
        color="fg.muted"
        size="xs"
        variant="ghost"
        _hover={{ color: 'fg.default' }}
      >
        <PiCaretDownBold />
      </IconButton>
    </HStack>
  );
};
