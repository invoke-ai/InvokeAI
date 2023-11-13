import { Flex, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import SyncModelsButton from './SyncModelsButton';

export default function SyncModels() {
  const { t } = useTranslation();

  return (
    <Flex
      sx={{
        w: 'full',
        p: 4,
        borderRadius: 4,
        gap: 4,
        justifyContent: 'space-between',
        alignItems: 'center',
        bg: 'base.200',
        _dark: { bg: 'base.800' },
      }}
    >
      <Flex
        sx={{
          flexDirection: 'column',
          gap: 2,
        }}
      >
        <Text sx={{ fontWeight: 600 }}>{t('modelManager.syncModels')}</Text>
        <Text fontSize="sm" sx={{ _dark: { color: 'base.400' } }}>
          {t('modelManager.syncModelsDesc')}
        </Text>
      </Flex>
      <SyncModelsButton />
    </Flex>
  );
}
