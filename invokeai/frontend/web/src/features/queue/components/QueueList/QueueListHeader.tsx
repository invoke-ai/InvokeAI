import { Flex } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';
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
      <Flex
        w={COLUMN_WIDTHS.number}
        justifyContent="flex-end"
        alignItems="center"
      >
        <InvText variant="subtext">#</InvText>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.statusBadge} alignItems="center">
        <InvText variant="subtext">{t('queue.status')}</InvText>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.time} alignItems="center">
        <InvText variant="subtext">{t('queue.time')}</InvText>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.batchId} alignItems="center">
        <InvText variant="subtext">{t('queue.batch')}</InvText>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.fieldValues} alignItems="center">
        <InvText variant="subtext">{t('queue.batchFieldValues')}</InvText>
      </Flex>
    </Flex>
  );
};

export default memo(QueueListHeader);
