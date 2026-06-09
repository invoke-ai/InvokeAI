import { Flex, HStack, Icon, IconButton, Stack, Text } from '@chakra-ui/react';
import { PiDotsThreeBold, PiStackBold } from 'react-icons/pi';

import { WidgetPanelFrame } from '../../components/WidgetFrames';
import { WidgetFailureBoundary } from '../../components/WidgetFailureBoundary';
import { useWorkbench } from '../../WorkbenchContext';

export const LayersWidgetView = () => {
  const { activeProject } = useWorkbench();

  return (
    <WidgetFailureBoundary widgetId="layers">
      <WidgetPanelFrame region="right">
        <HStack justify="space-between">
          <Text fontSize="xs" fontWeight="700">
            Layers
          </Text>
          <IconButton aria-label="Layer options" color="fg.muted" size="2xs" variant="ghost">
            <PiDotsThreeBold />
          </IconButton>
        </HStack>
        <Flex
          align="center"
          borderWidth="1px"
          borderColor="border.subtle"
          borderStyle="dashed"
          color="fg.subtle"
          direction="column"
          gap="2"
          justify="center"
          minH="8rem"
          rounded="md"
          p="4"
        >
          <Icon as={PiStackBold} boxSize="6" />
          <Text fontSize="2xs" textAlign="center">
            {activeProject.canvas.layers.length} layers, {activeProject.canvas.stagingArea.pendingImageIds.length}{' '}
            staged images.
          </Text>
        </Flex>
        <Stack bg="bg.surface" borderWidth="1px" borderColor="border.subtle" gap="1" p="2" rounded="md">
          <Text color="fg.muted" fontSize="2xs" fontWeight="700" textTransform="uppercase">
            Project Contracts
          </Text>
          <Text color="fg.subtle" fontSize="2xs">
            Graph history: {activeProject.graphHistory.length}
          </Text>
          <Text color="fg.subtle" fontSize="2xs">
            Events: {activeProject.events.length}
          </Text>
          <Text color="fg.subtle" fontSize="2xs">
            Undo: {activeProject.undoRedo.past.length} / Redo: {activeProject.undoRedo.future.length}
          </Text>
        </Stack>
      </WidgetPanelFrame>
    </WidgetFailureBoundary>
  );
};
