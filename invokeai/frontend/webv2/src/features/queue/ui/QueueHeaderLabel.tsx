import { Stack, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { useQueueCounts } from './queueDataStore';

/**
 * The Queue widget's header title: "Queue" over a live "N generating · M waiting"
 * summary. Rendered as the manifest `label`, so it sits in the standard frame
 * header. Counts come straight from the server-wide status, with tabular
 * numerals so the summary doesn't jitter as items move through the queue.
 */
export const QueueHeaderLabel = () => {
  const { t } = useTranslation();
  const counts = useQueueCounts();
  const isGenerating = counts.inProgress > 0;

  return (
    <Stack gap="0.5" minW="0">
      <Text fontSize="xs" fontWeight="700" lineHeight="1.15">
        {t('widgets.labels.queue')}
      </Text>
      <Text color="fg.subtle" fontSize="2xs" fontVariantNumeric="tabular-nums" lineHeight="1.15" truncate mb="-1.5">
        <Text as="span" color={isGenerating ? 'accent.solid' : 'fg.subtle'} fontWeight={isGenerating ? '600' : '400'}>
          {t('widgets.queue.generating', { count: counts.inProgress })}
        </Text>{' '}
        · {t('widgets.queue.waiting', { count: counts.pending })}
      </Text>
    </Stack>
  );
};
