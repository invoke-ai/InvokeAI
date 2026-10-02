import type { StarterModelBundle } from '@features/models/core/types';

import { HStack, Icon, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui';
import { DownloadIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** Summary + install action for the currently selected bundle. */
export const SelectedBundleBar = ({
  bundle,
  isInstalling,
  onInstall,
}: {
  bundle: StarterModelBundle;
  isInstalling: boolean;
  onInstall: () => void;
}) => {
  const { t } = useTranslation();
  const missingCount = bundle.models.filter((model) => !model.is_installed).length;

  return (
    <HStack gap="2" justify="space-between" px="3">
      <Text color="fg.subtle" fontSize="2xs" truncate>
        {bundle.name} · {t('models.modelCount', { count: bundle.models.length })} ·{' '}
        {missingCount === 0 ? t('models.allInstalled') : t('models.countToInstall', { count: missingCount })}
      </Text>
      {missingCount > 0 ? (
        <Button flexShrink={0} loading={isInstalling} size="2xs" variant="outline" onClick={onInstall}>
          <Icon as={DownloadIcon} boxSize="3" />
          {t('models.installBundle')}
        </Button>
      ) : null}
    </HStack>
  );
};
