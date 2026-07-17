import type { ReactNode } from 'react';

import { DataList, Separator, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import type { QueueServerItem } from './queueServerApi';

import { extractGenerationMeta } from './fieldValues';
import { formatDuration } from './formatDuration';
import { QueueItemActions } from './QueueItemActions';
import { useLocalGenerateValues } from './useLocalGenerateValues';

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
export const QueueItemDetails = ({ item }: { item: QueueServerItem }) => {
  const { t } = useTranslation();
  const meta = extractGenerationMeta(item);
  const localGenerateValues = useLocalGenerateValues(item.origin);
  const duration = formatDuration(item.started_at, item.completed_at);

  return (
    <DataList.Root gap="1.5" orientation="horizontal" size="sm">
      <DetailRow label={t('common.prompt')}>{meta.positivePrompt ?? '—'}</DetailRow>
      <DetailRow label={t('common.negative')}>{meta.negativePrompt ?? '—'}</DetailRow>
      <DetailRow label={t('common.seed')}>
        <Text as="span" fontVariantNumeric="tabular-nums">
          {meta.seed ?? '—'}
        </Text>
      </DetailRow>
      <DetailRow label={t('common.created')}>{new Date(item.created_at).toLocaleString()}</DetailRow>
      <DetailRow label={t('widgets.queue.took')}>{duration ?? '—'}</DetailRow>
      <DetailRow label={t('widgets.queue.batch')}>
        <Text as="span" truncate>
          {item.batch_id}
        </Text>
      </DetailRow>
      <DetailRow label={t('common.item')}>
        <Text as="span" fontVariantNumeric="tabular-nums">
          #{item.item_id}
        </Text>
      </DetailRow>
      {item.error_message ? (
        <DetailRow label={t('common.error')}>
          <Text as="span" color="fg.error">
            {item.error_message}
          </Text>
        </DetailRow>
      ) : null}

      <Separator borderColor="border.subtle" my="0.5" />
      <QueueItemActions item={item} localGenerateValues={localGenerateValues} />
    </DataList.Root>
  );
};
