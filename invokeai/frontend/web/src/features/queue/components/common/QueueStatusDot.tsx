import { Box } from '@chakra-ui/react';
import { memo, useMemo } from 'react';
import { SessionQueueItemStatus } from 'services/api/endpoints/queue';

const STATUSES = {
  pending: { colorScheme: 'cyan', translationKey: 'queue.pending' },
  in_progress: { colorScheme: 'yellow', translationKey: 'queue.in_progress' },
  completed: { colorScheme: 'green', translationKey: 'queue.completed' },
  failed: { colorScheme: 'red', translationKey: 'queue.failed' },
  canceled: { colorScheme: 'orange', translationKey: 'queue.canceled' },
};

const QueueStatusDot = ({ status }: { status: SessionQueueItemStatus }) => {
  const sx = useMemo(
    () => ({
      w: 2,
      h: 2,
      bg: `${STATUSES[status].colorScheme}.${500}`,
      _dark: {
        bg: `${STATUSES[status].colorScheme}.${400}`,
      },
      borderRadius: '100%',
    }),
    [status]
  );
  return <Box sx={sx} />;
};
export default memo(QueueStatusDot);
