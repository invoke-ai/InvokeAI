import type { StarterModelBundle } from '@features/models/core/types';

import { HStack, Icon, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
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
    <HStack gap="2" justify="space-between" px="3" wrap="wrap">
      <Text color="fg.muted" fontSize="xs" minW="0">
        {t('models.modelCount', { count: bundle.models.length })} ·{' '}
        {missingCount === 0 ? t('models.allInstalled') : t('models.countToInstall', { count: missingCount })}
      </Text>
      {missingCount > 0 ? (
        <Button flexShrink={0} loading={isInstalling} size="sm" variant="solid" onClick={onInstall}>
          <Icon as={DownloadIcon} boxSize="3" />
          {t('models.installNamedBundle', { name: bundle.name })}
        </Button>
      ) : null}
    </HStack>
  );
};
