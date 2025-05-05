import { Flex, Text } from '@invoke-ai/ui-library';
import { selectShouldShowCredits } from 'features/system/store/configSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useSelector } from 'react-redux';

import { COLUMN_WIDTHS } from './constants';

const QueueListHeader = () => {
  const { t } = useTranslation();
  const shouldShowCredits = useSelector(selectShouldShowCredits);
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
      <Flex ps={0.5} w={COLUMN_WIDTHS.origin} alignItems="center">
        <Text variant="subtext">{t('queue.origin')}</Text>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.destination} alignItems="center">
        <Text variant="subtext">{t('queue.destination')}</Text>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.time} alignItems="center">
        <Text variant="subtext">{t('queue.time')}</Text>
      </Flex>
      {shouldShowCredits && (
        <Flex ps={0.5} w={COLUMN_WIDTHS.credits} alignItems="center">
          <Text variant="subtext">{t('queue.credits')}</Text>
        </Flex>
      )}
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
