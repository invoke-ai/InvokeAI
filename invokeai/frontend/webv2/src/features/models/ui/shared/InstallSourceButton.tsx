/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import { Badge, HStack, Icon, Spinner } from '@chakra-ui/react';
import { openInstallQueue, openModelDetail } from '@features/models/ui/uiStore';
import { Button } from '@platform/ui';
import { DownloadIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { useInstallSourceStage } from './InstallSourceMenu';

export const InstallSourceButton = ({
  installedModelKey = null,
  isInstalled = false,
  isPending = false,
  name,
  onInstall,
  source,
}: {
  /** The library model this source landed as; implies installed. */
  installedModelKey?: string | null;
  /** Already in the library, without a known model to open. */
  isInstalled?: boolean;
  /** The install POST is in flight (before a job exists). */
  isPending?: boolean;
  /** What installs, so repeated Install buttons stay distinguishable to assistive tech. */
  name: string;
  onInstall: () => void;
  /** Source string used to match an active install job. */
  source: string;
}) => {
  const { t } = useTranslation();
  const stage = useInstallSourceStage(source, { installedModelKey, isInstalled, isPending });

  if (stage.kind === 'installed') {
    const { modelKey } = stage;

    return (
      <HStack flexShrink={0} gap="1.5">
        <Badge colorPalette="green" fontSize="xs" size="lg" variant="surface">
          {t('models.installed')}
        </Badge>
        {modelKey !== null ? (
          <Button size="sm" variant="ghost" onClick={() => openModelDetail(modelKey)}>
            {t('models.viewModel')}
          </Button>
        ) : null}
      </HStack>
    );
  }

  if (stage.kind === 'installing') {
    return (
      <HStack flexShrink={0} gap="1.5">
        <Badge colorPalette="blue" fontSize="xs" size="lg" variant="surface">
          <Spinner borderWidth="1.5px" boxSize="2.5" />
          {t('models.installing')}
        </Badge>
        <Button size="sm" variant="ghost" onClick={openInstallQueue}>
          {t('models.viewQueue')}
        </Button>
      </HStack>
    );
  }

  return (
    <Button aria-label={t('models.installNamed', { name })} size="sm" variant="outline" onClick={onInstall}>
      <Icon as={DownloadIcon} boxSize="3" />
      {t('models.install')}
    </Button>
  );
};
