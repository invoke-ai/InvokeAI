import { Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { COLUMN_WIDTHS } from './constants';
import QueueListHeaderColumn from './QueueListHeaderColumn';

const QueueListHeader = () => {
  const { t } = useTranslation();

  return (
    <Flex
      alignItems="center"
      gap={4}
      p={1}
      pb={2}
      textTransform="uppercase"
      fontWeight="bold"
      fontSize="sm"
      letterSpacing={1}
    >
      <QueueListHeaderColumn displayName="#" w={COLUMN_WIDTHS.number} alignItems="center" />
      <QueueListHeaderColumn
        field="created_at"
        displayName={t('queue.createdAt')}
        ps={0.5}
        w={COLUMN_WIDTHS.createdAt}
        alignItems="center"
      />
      <QueueListHeaderColumn
        displayName={t('queue.status')}
        ps={0.5}
        w={COLUMN_WIDTHS.statusBadge}
        alignItems="center"
      />

      <QueueListHeaderColumn displayName={t('queue.time')} ps={0.5} w={COLUMN_WIDTHS.time} alignItems="center" />
      <QueueListHeaderColumn
        displayName={`${t('queue.origin')} / ${t('queue.destination')}`}
        ps={0.5}
        w={COLUMN_WIDTHS.origin_destination}
        alignItems="center"
      />
      <QueueListHeaderColumn displayName={t('queue.batch')} ps={0.5} w={COLUMN_WIDTHS.batchId} alignItems="center" />
      <QueueListHeaderColumn displayName={t('queue.user')} ps={0.5} w={COLUMN_WIDTHS.user} alignItems="center" />
      <QueueListHeaderColumn
        displayName={t('queue.batchFieldValues')}
        ps={0.5}
        w={COLUMN_WIDTHS.fieldValues}
        alignItems="center"
        flexGrow={1}
      />
    </Flex>
  );
};

export default memo(QueueListHeader);
