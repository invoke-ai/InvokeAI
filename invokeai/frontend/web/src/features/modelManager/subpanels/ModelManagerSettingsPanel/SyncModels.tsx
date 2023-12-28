import { Flex } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';
import { SyncModelsButton } from 'features/modelManager/components/SyncModels/SyncModelsButton';
import { useTranslation } from 'react-i18next';

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
        bg: 'base.800',
      }}
    >
      <Flex
        sx={{
          flexDirection: 'column',
          gap: 2,
        }}
      >
        <InvText sx={{ fontWeight: 'semibold' }}>
          {t('modelManager.syncModels')}
        </InvText>
        <InvText fontSize="sm" sx={{ color: 'base.400' }}>
          {t('modelManager.syncModelsDesc')}
        </InvText>
      </Flex>
      <SyncModelsButton />
    </Flex>
  );
}
