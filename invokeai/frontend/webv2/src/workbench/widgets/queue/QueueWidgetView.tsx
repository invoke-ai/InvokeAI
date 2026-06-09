import { Stack, Text } from '@chakra-ui/react';
import { PiListNumbersBold } from 'react-icons/pi';

import { StatusWidgetChip, WidgetPanelFrame } from '../../components/WidgetFrames';
import { WidgetFailureBoundary } from '../../components/WidgetFailureBoundary';
import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';

export const QueueWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const { activeProject } = useWorkbench();
  const pendingQueueCount = activeProject.queue.items.filter((item) => item.status === 'pending').length;

  if (region === 'bottom' && presentation !== 'expanded') {
    return <StatusWidgetChip icon={PiListNumbersBold}>{pendingQueueCount} queued</StatusWidgetChip>;
  }

  if (region === 'bottom') {
    return <QueueContents />;
  }

  return (
    <WidgetFailureBoundary widgetId="queue">
      <WidgetPanelFrame region="right">
        <Text fontSize="xs" fontWeight="700">
          Queue
        </Text>
        <QueueContents />
      </WidgetPanelFrame>
    </WidgetFailureBoundary>
  );
};

const QueueContents = () => {
  const { activeProject } = useWorkbench();

  return (
    <Stack gap="1">
      {activeProject.queue.items.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          Queue submissions will appear here.
        </Text>
      ) : (
        activeProject.queue.items.map((item) => (
          <Text key={item.id} color="fg.subtle" fontSize="2xs">
            {item.id}: {item.status}
          </Text>
        ))
      )}
    </Stack>
  );
};
