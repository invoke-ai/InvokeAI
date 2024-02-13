import { Flex, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { COLUMN_WIDTHS } from './constants';
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
      <Flex w={COLUMN_WIDTHS.number} justifyContent="flex-end" alignItems="center">
        <Text variant="subtext">#</Text>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.statusBadge} alignItems="center">
        <Text variant="subtext">{t('queue.status')}</Text>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.time} alignItems="center">
        <Text variant="subtext">{t('queue.time')}</Text>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.batchId} alignItems="center">
        <Text variant="subtext">{t('queue.batch')}</Text>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.fieldValues} alignItems="center">
        <Text variant="subtext">{t('queue.batchFieldValues')}</Text>
      </Flex>
    </Flex>
  );
};

export default memo(QueueListHeader);
