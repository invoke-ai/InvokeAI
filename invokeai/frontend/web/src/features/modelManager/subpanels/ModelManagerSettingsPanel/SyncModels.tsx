import { Flex, Text } from '@invoke-ai/ui-library';
import { SyncModelsButton } from 'features/modelManager/components/SyncModels/SyncModelsButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const SyncModels = () => {
  const { t } = useTranslation();

  return (
    <Flex w="full" p={4} borderRadius={4} gap={4} justifyContent="space-between" alignItems="center" bg="base.800">
      <Flex flexDirection="column" gap={2}>
        <Text fontWeight="semibold">{t('modelManager.syncModels')}</Text>
        <Text fontSize="sm" variant="subtext">
          {t('modelManager.syncModelsDesc')}
        </Text>
      </Flex>
      <SyncModelsButton />
    </Flex>
  );
};

export default memo(SyncModels);
