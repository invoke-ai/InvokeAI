import { Box, Flex, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { Panel } from '@workbench/components/ui';
import { useActiveProject } from '@workbench/WorkbenchContext';
import { LayersIcon } from 'lucide-react';

export const LayersWidgetView = () => {
  const activeProject = useActiveProject();
  const layers = activeProject.canvas.document.layers;

  return (
    <>
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
        <Icon as={LayersIcon} boxSize="6" />
        <Text fontSize="2xs" textAlign="center">
          {layers.length} layers, {activeProject.canvas.stagingArea.pendingImageIds.length} staged images.
        </Text>
      </Flex>
      <Panel gap="2" p="2">
        {layers.length === 0 ? (
          <Text color="fg.subtle" fontSize="2xs">
            Accepted canvas layers will appear here.
          </Text>
        ) : (
          layers.map((layer) => (
            <HStack key={layer.id} gap="2">
              <Box
                bg="bg.emphasized"
                borderWidth="1px"
                borderColor="border.subtle"
                h="10"
                overflow="hidden"
                rounded="sm"
                w="10"
              >
                <img
                  alt={layer.label}
                  src={layer.thumbnailUrl || layer.imageUrl}
                  style={{ height: '100%', objectFit: 'cover', width: '100%' }}
                />
              </Box>
              <Stack gap="0" minW="0">
                <Text fontSize="2xs" fontWeight="700" truncate>
                  {layer.label}
                </Text>
                <Text color="fg.subtle" fontSize="2xs" truncate>
                  {layer.width} x {layer.height}
                </Text>
              </Stack>
            </HStack>
          ))
        )}
      </Panel>
      <Panel gap="1" p="2">
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
      </Panel>
    </>
  );
};
