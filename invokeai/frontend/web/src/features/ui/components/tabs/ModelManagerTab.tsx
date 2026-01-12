import { Center, Flex, Heading, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { ModelManager } from 'features/modelManagerV2/subpanels/ModelManager';
import { ModelPane } from 'features/modelManagerV2/subpanels/ModelPane';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ModelManagerTab = () => {
  const user = useAppSelector(selectCurrentUser);
  const { t } = useTranslation();

  // Show access denied message for non-admin users
  if (!user?.is_admin) {
    return (
      <Flex layerStyle="body" w="full" h="full" p={4}>
        <Center w="full" h="full" flexDir="column" gap={4}>
          <Heading size="lg">{t('modelManager.modelManager')}</Heading>
          <Text fontSize="md" color="base.500">
            {t('auth.adminOnlyFeature')}
          </Text>
        </Center>
      </Flex>
    );
  }

  return (
    <Flex layerStyle="body" w="full" h="full" gap="2" p={2}>
      <ModelManager />
      <ModelPane />
    </Flex>
  );
};

export default memo(ModelManagerTab);
