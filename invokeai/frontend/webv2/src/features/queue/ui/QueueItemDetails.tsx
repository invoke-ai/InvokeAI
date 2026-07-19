import type { QueueItemReadModel } from '@features/queue/core/types';
import type { ReactNode } from 'react';

import { DataList, Separator, Text } from '@chakra-ui/react';
import { extractGenerationMeta } from '@features/queue/core/generationMeta';
import { useTranslation } from 'react-i18next';

import { formatDuration } from './formatDuration';
import { useQueueUi } from './QueueUiContext';

const DetailRow = ({ label, children }: { label: string; children: ReactNode }) => (
  <DataList.Item key={label} alignItems="start">
    <DataList.ItemLabel>{label}</DataList.ItemLabel>
    {/* minW=0 lets unbreakable values (batch UUIDs) truncate instead of
        pushing the row's min-content past the panel. */}
    <DataList.ItemValue fontFamily="mono" fontSize="2xs" minW="0">
      {children}
    </DataList.ItemValue>
  </DataList.Item>
);

/** Expanded detail grid + actions for a RECENT queue item row. */
export const QueueItemDetails = ({ item }: { item: QueueItemReadModel }) => {
  const { t } = useTranslation();
  const { ItemActions } = useQueueUi();
  const meta = extractGenerationMeta(item);
  const duration = formatDuration(item.startedAt, item.completedAt);

  return (
    <DataList.Root gap="1.5" orientation="horizontal" size="sm">
      <DetailRow label={t('common.prompt')}>{meta.positivePrompt ?? '—'}</DetailRow>
      <DetailRow label={t('common.negative')}>{meta.negativePrompt ?? '—'}</DetailRow>
      <DetailRow label={t('common.seed')}>
        <Text as="span" fontVariantNumeric="tabular-nums">
          {meta.seed ?? '—'}
        </Text>
      </DetailRow>
      <DetailRow label={t('common.created')}>{new Date(item.createdAt).toLocaleString()}</DetailRow>
      <DetailRow label={t('widgets.queue.took')}>{duration ?? '—'}</DetailRow>
      <DetailRow label={t('widgets.queue.batch')}>
        <Text as="span" truncate>
          {item.batchId}
        </Text>
      </DetailRow>
      <DetailRow label={t('common.item')}>
        <Text as="span" fontVariantNumeric="tabular-nums">
          #{item.id}
        </Text>
      </DetailRow>
      {item.errorMessage ? (
        <DetailRow label={t('common.error')}>
          <Text as="span" color="fg.error">
            {item.errorMessage}
          </Text>
        </DetailRow>
      ) : null}

      <Separator borderColor="border.subtle" my="0.5" />
      <ItemActions item={item} />
    </DataList.Root>
  );
};
