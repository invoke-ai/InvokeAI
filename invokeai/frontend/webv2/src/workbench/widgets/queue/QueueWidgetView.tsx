import { Badge, Stack, Text, HStack } from '@chakra-ui/react';
import { PiListNumbersBold } from 'react-icons/pi';

import { StatusWidgetChip } from '../../components/WidgetFrames';
import { Button } from '../../components/ui/Button';
import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';
import { getDestinationLabel, getSourceLabel } from '../../invocation';

export const QueueWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const { activeProject } = useWorkbench();
  const pendingQueueCount = activeProject.queue.items.filter(
    (item) => item.status === 'pending' || item.status === 'running'
  ).length;

  if (region === 'bottom' && presentation !== 'expanded') {
    return <StatusWidgetChip icon={PiListNumbersBold}>{pendingQueueCount} queued</StatusWidgetChip>;
  }

  if (region === 'bottom') {
    return <QueueContents />;
  }

  return <QueueContents />;
};

const QueueContents = () => {
  const { dispatch, state } = useWorkbench();
  const queueRows = state.projects.flatMap((project) =>
    project.queue.items.map((item) => ({ item, projectName: project.name }))
  );

  return (
    <Stack gap="2">
      {queueRows.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          Queue submissions will appear here.
        </Text>
      ) : (
        queueRows.map(({ item, projectName }) => {
          const backendIds = item.backendItemIds?.length ? ` backend ${item.backendItemIds.join(', ')}` : '';
          const resultCount = item.resultImages?.length ? `${item.resultImages.length} image(s)` : 'No results yet';
          const generateValues = item.snapshot.widgetStates.generate.values as Record<string, unknown>;
          const prompt = typeof generateValues.positivePrompt === 'string' ? generateValues.positivePrompt.trim() : '';
          const promptSummary = prompt ? prompt.slice(0, 72) : 'No prompt snapshot';
          const canCancel = item.cancellable && (item.status === 'pending' || item.status === 'running');

          return (
            <Stack
              key={item.id}
              bg="bg.surface"
              borderWidth="1px"
              borderColor={item.status === 'failed' ? 'red.500' : 'border.subtle'}
              gap="1"
              p="2"
              rounded="md"
            >
              <HStack justify="space-between">
                <Text fontSize="2xs" fontWeight="700">
                  {projectName}
                </Text>
                <Badge colorPalette={item.status === 'failed' ? 'red' : item.status === 'completed' ? 'green' : 'blue'}>
                  {item.status}
                </Badge>
              </HStack>
              <Text color="fg.subtle" fontSize="2xs">
                {getSourceLabel(item.snapshot.sourceId)} to {getDestinationLabel(item.snapshot.destination)} · Graph{' '}
                {item.snapshot.graph.label || item.snapshot.graph.id}
              </Text>
              <Text color="fg.subtle" fontSize="2xs">
                {promptSummary}
              </Text>
              <HStack justify="space-between">
                <Text color={item.status === 'failed' ? 'red.300' : 'fg.subtle'} fontSize="2xs">
                  {item.id}
                  {backendIds} · {resultCount}
                  {item.error ? ` · ${item.error}` : ''}
                </Text>
                {canCancel ? (
                  <Button
                    size="2xs"
                    variant="outline"
                    onClick={() => dispatch({ queueItemId: item.id, type: 'cancelQueueItem' })}
                  >
                    Cancel
                  </Button>
                ) : null}
              </HStack>
            </Stack>
          );
        })
      )}
    </Stack>
  );
};
