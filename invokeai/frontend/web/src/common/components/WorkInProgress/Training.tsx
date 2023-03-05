import { Flex, Heading, Text, VStack } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import WorkInProgress from './WorkInProgress';

export default function TrainingWIP() {
  const { t } = useTranslation();
  return (
    <WorkInProgress>
      <Flex
        sx={{
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          w: '100%',
          h: '100%',
          gap: 4,
          textAlign: 'center',
        }}
      >
        <Heading>{t('common.training')}</Heading>
        <VStack maxW="50rem" gap={4}>
          <Text>{t('common.trainingDesc1')}</Text>
          <Text>{t('common.trainingDesc2')}</Text>
        </VStack>
      </Flex>
    </WorkInProgress>
  );
}
