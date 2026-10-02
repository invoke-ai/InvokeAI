import { Stack, Text } from '@chakra-ui/react';
import { getPersonalQueueActivity } from '@features/queue/core/types';
import { useTranslation } from 'react-i18next';

import { useQueueCounts } from './queueDataStore';

/** Use server-wide queue counts and tabular numerals to avoid header jitter. */
export const QueueHeaderLabel = () => {
  const { t } = useTranslation();
  const counts = useQueueCounts();
  const activity = getPersonalQueueActivity(counts);
  const isGenerating = activity.inProgress > 0;

  return (
    <Stack gap="0.5" minW="0">
      <Text fontSize="xs" fontWeight="700" lineHeight="1.15">
        {t('widgets.labels.queue')}
      </Text>
      <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums" lineHeight="1.15" truncate mb="-1.5">
        <Text as="span" color={isGenerating ? 'accent.solid' : 'fg.muted'} fontWeight={isGenerating ? '600' : '400'}>
          {t('widgets.queue.generating', { count: activity.inProgress })}
        </Text>{' '}
        · {t('widgets.queue.waiting', { count: activity.pending })}
      </Text>
    </Stack>
  );
};
