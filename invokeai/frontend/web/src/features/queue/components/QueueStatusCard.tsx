import { Flex, Heading, Text } from '@chakra-ui/react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const QueueStatusCard = () => {
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();

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
        {queueStatus?.queue.pending}
      </Text>
      <Text>
        <Text as="span" fontWeight={600}>
          {t('queue.in_progress')}:{' '}
        </Text>
        {queueStatus?.queue.in_progress}
      </Text>
      <Text>
        <Text as="span" fontWeight={600}>
          {t('queue.completed')}:{' '}
        </Text>
        {queueStatus?.queue.completed}
      </Text>
      <Text>
        <Text as="span" fontWeight={600}>
          {t('queue.failed')}:{' '}
        </Text>
        {queueStatus?.queue.failed}
      </Text>
      <Text>
        <Text as="span" fontWeight={600}>
          {t('queue.canceled')}:{' '}
        </Text>
        {queueStatus?.queue.canceled}
      </Text>
    </Flex>
  );
};

export default memo(QueueStatusCard);
