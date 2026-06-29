import type { BackendQueueItemStatus } from '@workbench/backend/events';

import { Box } from '@chakra-ui/react';

import { getStatusMeta } from './statusMeta';

const ACTIVE_HALO = { boxShadow: '0 0 0 3px {colors.accent.subtle}' } as const;

/** Small status indicator dot; the running state gets a soft accent halo. */
export const QueueStatusDot = ({ status, boxSize = '2' }: { status: BackendQueueItemStatus; boxSize?: string }) => {
  const { dotColor, label } = getStatusMeta(status);

  return (
    <Box
      aria-label={label}
      bg={dotColor}
      boxSize={boxSize}
      flexShrink={0}
      rounded="full"
      role="img"
      css={status === 'in_progress' ? ACTIVE_HALO : undefined}
    />
  );
};
