import { Box, Flex, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { Panel } from '@workbench/components/ui';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { LayersIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const LayersWidgetView = () => {
  const { t } = useTranslation();
  const { eventsCount, graphHistoryCount, layers, redoCount, stagedImageCount, undoCount } = useActiveProjectSelector(
    (project) => ({
      eventsCount: project.events.length,
      graphHistoryCount: project.graphHistory.length,
      layers: project.canvas.document.layers,
      redoCount: project.undoRedo.future.length,
      stagedImageCount: project.canvas.stagingArea.pendingImageIds.length,
      undoCount: project.undoRedo.past.length,
    }),
    (left, right) =>
      left.eventsCount === right.eventsCount &&
      left.graphHistoryCount === right.graphHistoryCount &&
      left.layers === right.layers &&
      left.redoCount === right.redoCount &&
      left.stagedImageCount === right.stagedImageCount &&
      left.undoCount === right.undoCount
  );

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
          {t('widgets.layers.summary', { layers: layers.length, stagedImages: stagedImageCount })}
        </Text>
      </Flex>
      <Panel gap="2" p="2">
        {layers.length === 0 ? (
          <Text color="fg.subtle" fontSize="2xs">
            {t('widgets.layers.empty')}
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
          {t('widgets.layers.projectContracts')}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.layers.graphHistoryCount', { count: graphHistoryCount })}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.layers.eventsCount', { count: eventsCount })}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.layers.undoRedoCount', { redo: redoCount, undo: undoCount })}
        </Text>
      </Panel>
    </>
  );
};
