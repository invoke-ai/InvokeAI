import type { WidgetViewProps } from '@workbench/types';

import { Badge, HStack, Progress, Stack, Text } from '@chakra-ui/react';
import { useQueueItemProgress } from '@workbench/backend/progressStore';
import { Button } from '@workbench/components/ui/Button';
import { StatusWidgetChip } from '@workbench/components/WidgetFrames';
import { getDestinationLabel, getSourceLabel } from '@workbench/invocation';
import { useActiveProjectSelector, useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { ListOrderedIcon } from 'lucide-react';

export const QueueWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const pendingQueueCount = queueItems.filter((item) => item.status === 'pending' || item.status === 'running').length;

  if (region === 'bottom' && presentation !== 'expanded') {
    return <StatusWidgetChip icon={ListOrderedIcon}>{pendingQueueCount} queued</StatusWidgetChip>;
  }

  if (region === 'bottom') {
    return <QueueContents />;
  }

  return <QueueContents />;
};

const QueueItemLiveProgress = ({ queueItemId }: { queueItemId: string }) => {
  const progress = useQueueItemProgress(queueItemId);

  if (!progress || (!progress.message && progress.percentage === null)) {
    return null;
  }

  return (
    <Stack gap="1">
      <Progress.Root
        aria-label={progress.message || 'Generation progress'}
        max={1}
        size="xs"
        value={progress.percentage}
      >
        <Progress.Track>
          <Progress.Range />
        </Progress.Track>
      </Progress.Root>
      {progress.message ? (
        <Text color="fg.subtle" fontSize="2xs">
          {progress.message}
        </Text>
      ) : null}
    </Stack>
  );
};

const QueueContents = () => {
  const projects = useWorkbenchSelector((snapshot) => snapshot.state.projects);
  const dispatch = useWorkbenchDispatch();
  const queueRows = projects.flatMap((project) =>
    project.queue.items.map((item) => ({ item, projectId: project.id, projectName: project.name }))
  );

  return (
    <Stack gap="2">
      {queueRows.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          Queue submissions will appear here.
        </Text>
      ) : (
        queueRows.map(({ item, projectId, projectName }) => {
          const backendIds = item.backendItemIds?.length ? ` backend ${item.backendItemIds.join(', ')}` : '';
          const resultCount = item.resultImages?.length ? `${item.resultImages.length} image(s)` : 'No results yet';
          const generateValues = item.snapshot.widgetStates.generate.values as Record<string, unknown>;
          const prompt = typeof generateValues.positivePrompt === 'string' ? generateValues.positivePrompt.trim() : '';
          const promptSummary = prompt ? prompt.slice(0, 72) : 'No prompt snapshot';
          const canCancel = item.cancellable && (item.status === 'pending' || item.status === 'running');

          return (
            <Stack
              key={item.id}
              bg="bg.subtle"
              borderWidth="1px"
              borderColor={item.status === 'failed' ? 'fg.error' : 'border.subtle'}
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
              {item.status === 'running' ? <QueueItemLiveProgress queueItemId={item.id} /> : null}
              <HStack justify="space-between">
                <Text color={item.status === 'failed' ? 'fg.error' : 'fg.subtle'} fontSize="2xs">
                  {item.id}
                  {backendIds} · {resultCount}
                  {item.error ? ` · ${item.error}` : ''}
                </Text>
                {canCancel ? (
                  <Button
                    size="2xs"
                    variant="outline"
                    onClick={() => dispatch({ projectId, queueItemId: item.id, type: 'cancelQueueItem' })}
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
