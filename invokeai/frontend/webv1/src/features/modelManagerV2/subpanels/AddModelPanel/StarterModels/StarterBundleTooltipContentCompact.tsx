import { Flex, Text } from '@invoke-ai/ui-library';
import { useStarterBundleInstallStatus } from 'features/modelManagerV2/hooks/useStarterBundleInstallStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { S } from 'services/api/types';

export const StarterBundleTooltipContentCompact = memo(({ bundle }: { bundle: S['StarterModelBundle'] }) => {
  const { t } = useTranslation();
  const { total, install, skip } = useStarterBundleInstallStatus(bundle);

  return (
    <Flex flexDir="column" gap={1} p={1}>
      <Text>{t('modelManager.includesNModels', { n: total })}</Text>
      {install.length === 0 && (
        <Text fontWeight="semibold">{t('modelManager.allNModelsInstalled', { count: total })}.</Text>
      )}
      {install.length > 0 && (
        <Text fontWeight="semibold">{t('modelManager.installedModelsCount', { installed: skip.length, total })}</Text>
      )}
    </Flex>
  );
});
StarterBundleTooltipContentCompact.displayName = 'StarterBundleTooltipContentCompact';
