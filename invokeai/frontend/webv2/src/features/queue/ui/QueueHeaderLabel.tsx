import { Stack, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { useQueueWidgetActivity } from './useQueueWidgetActivity';

/** Use server-wide queue counts and tabular numerals to avoid header jitter. */
export const QueueHeaderLabel = () => {
  const { t } = useTranslation();
  const { activity, inProgress, loadingModel, pending } = useQueueWidgetActivity();
  const isGenerating = activity === 'running' || activity === 'loading-model';
  // What holds the work, when something does: the connection, the processor, or a model still loading.
  const state =
    activity === 'offline'
      ? t('widgets.queue.reconnecting')
      : activity === 'paused'
        ? t('widgets.queue.processorPaused')
        : activity === 'loading-model'
          ? t('widgets.queue.loadingModel', { model: loadingModel })
          : null;

  return (
    <Stack gap="0.5" minW="0">
      <Text fontWeight="700" lineHeight="1.15">
        {t('widgets.labels.queue')}
      </Text>
      <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums" lineHeight="1.15" truncate mb="-1.5">
        <Text as="span" color={isGenerating ? 'accent.solid' : 'fg.muted'} fontWeight={isGenerating ? '600' : '400'}>
          {t('widgets.queue.generating', { count: inProgress })}
        </Text>{' '}
        · {t('widgets.queue.waiting', { count: pending })}
        {state ? (
          <Text as="span" color={activity === 'loading-model' ? 'fg.muted' : 'fg.warning'}>
            {' '}
            · {state}
          </Text>
        ) : null}
      </Text>
    </Stack>
  );
};
