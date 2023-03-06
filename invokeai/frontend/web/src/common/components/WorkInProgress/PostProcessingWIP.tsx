import { Flex, Heading, Text, VStack } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import WorkInProgress from './WorkInProgress';

export const PostProcessingWIP = () => {
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
        <Heading>{t('common.postProcessing')}</Heading>
        <VStack maxW="50rem" gap={4}>
          <Text>{t('common.postProcessDesc1')}</Text>
          <Text>{t('common.postProcessDesc2')}</Text>
          <Text>{t('common.postProcessDesc3')}</Text>
        </VStack>
      </Flex>
    </WorkInProgress>
  );
};
