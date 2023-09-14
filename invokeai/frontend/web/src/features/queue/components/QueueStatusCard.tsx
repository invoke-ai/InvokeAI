import { Flex, Heading, Text } from '@chakra-ui/react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const QueueStatusCard = () => {
  const { t } = useTranslation();
  const { data: queueStatusData } = useGetQueueStatusQuery();

  return (
    <Flex
      layerStyle="second"
      borderRadius="base"
      p={2}
      flexDir="column"
      gap={1}
      w={96}
    >
      <Heading size="md">{t('queue.status')}</Heading>
      <Text>
        <Text as="span" fontWeight={600}>
          {t('queue.pending')}:{' '}
        </Text>
        {queueStatusData?.pending}
      </Text>
      <Text>
        <Text as="span" fontWeight={600}>
          {t('queue.inProgress')}:{' '}
        </Text>
        {queueStatusData?.in_progress}
      </Text>
      <Text>
        <Text as="span" fontWeight={600}>
          {t('queue.completed')}:{' '}
        </Text>
        {queueStatusData?.completed}
      </Text>
      <Text>
        <Text as="span" fontWeight={600}>
          {t('queue.failed')}:{' '}
        </Text>
        {queueStatusData?.failed}
      </Text>
      <Text>
        <Text as="span" fontWeight={600}>
          {t('queue.canceled')}:{' '}
        </Text>
        {queueStatusData?.canceled}
      </Text>
    </Flex>
  );
};

export default memo(QueueStatusCard);
