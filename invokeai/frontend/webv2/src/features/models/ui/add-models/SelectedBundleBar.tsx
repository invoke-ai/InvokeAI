import type { BundlePreviewDecoder } from '@features/models/core/starters';
import type { StarterModelBundle } from '@features/models/core/types';

import { HStack, Icon, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { ChevronRightIcon, DownloadIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** Summary + install action for the currently selected bundle. */
export const SelectedBundleBar = ({
  bundle,
  decoder,
  isInstalling,
  onInstall,
  onShowDecoder,
}: {
  bundle: StarterModelBundle;
  /** The family's preview decoder when it is missing; bundles never include one. */
  decoder: BundlePreviewDecoder | null;
  isInstalling: boolean;
  onInstall: () => void;
  onShowDecoder: () => void;
}) => {
  const { t } = useTranslation();
  const missingCount = bundle.models.filter((model) => !model.is_installed).length;

  return (
    <HStack gap="2" justify="space-between" px="3" wrap="wrap">
      <Text color="fg.muted" fontSize="xs" minW="0">
        {t('models.modelCount', { count: bundle.models.length })} ·{' '}
        {missingCount === 0 ? t('models.allInstalled') : t('models.countToInstall', { count: missingCount })}
        {decoder ? (
          <>
            {' · '}
            {t('models.previewDecoderHint')}{' '}
            <Button
              aria-label={t('models.showStarterModel', { name: decoder.decoder.name })}
              color="fg.muted"
              fontSize="xs"
              gap="0.5"
              h="auto"
              minW="0"
              p="0"
              size="sm"
              textDecoration="underline"
              variant="plain"
              onClick={onShowDecoder}
            >
              {decoder.decoder.name}
              <Icon as={ChevronRightIcon} boxSize="3" />
            </Button>
          </>
        ) : null}
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
