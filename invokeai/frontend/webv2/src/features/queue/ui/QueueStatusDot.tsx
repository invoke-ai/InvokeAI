import type { QueueItemStatus } from '@features/queue/core/types';

import { Box } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { getStatusMeta } from './statusMeta';

const ACTIVE_HALO = { boxShadow: '0 0 0 3px {colors.accent.subtle}' } as const;

/** Small status indicator dot; the running state gets a soft accent halo. */
export const QueueStatusDot = ({ status, boxSize = '2' }: { status: QueueItemStatus; boxSize?: string }) => {
  const { t } = useTranslation();
  const { dotColor, labelKey } = getStatusMeta(status);

  return (
    <Box
      aria-label={t(labelKey)}
      bg={dotColor}
      boxSize={boxSize}
      flexShrink={0}
      rounded="full"
      role="img"
      css={status === 'in_progress' ? ACTIVE_HALO : undefined}
    />
  );
};
