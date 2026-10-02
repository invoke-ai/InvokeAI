import type { StarterModelBundle } from '@features/models/core/types';

import { Flex, Icon, Spinner, Stack, Text, Wrap } from '@chakra-ui/react';
import { ensureInstallsLoaded, isActiveInstallStatus, useInstallsSelector } from '@features/models/data/installsStore';
import { ensureModelsLoaded, useModelsSelector } from '@features/models/data/modelsStore';
import { ensureStartersLoaded, useStartersSelector } from '@features/models/data/startersStore';
import { openAddModelsWithBundle } from '@features/models/ui/uiStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { useNavigate } from '@tanstack/react-router';
import { ArrowRightIcon, BoxIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Explain missing models during first-run setup or starter downloads; hide the notice once usable library content
 * exists.
 */

const EMPTY_BUNDLES: Record<string, StarterModelBundle> = {};

const PANEL_STYLE = {
  bg: 'bg.subtle',
  borderColor: 'border.subtle',
  borderWidth: '1px',
  gap: '3',
  p: '4',
  rounded: 'lg',
} as const;

export const ModelsNotice = () => {
  const { t } = useTranslation();
  const modelCount = useModelsSelector((snapshot) => snapshot.models.length);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const bundles = useStartersSelector((snapshot) => snapshot.response?.starter_bundles ?? EMPTY_BUNDLES);
  const activeInstallCount = useInstallsSelector(
    (snapshot) => snapshot.jobs.filter((job) => isActiveInstallStatus(job.status)).length
  );

  useMountEffect(() => {
    void ensureModelsLoaded();
    ensureStartersLoaded();
    ensureInstallsLoaded();
  });

  // Wait for authoritative library data before claiming no models are installed.
  if (modelsStatus !== 'loaded' || modelCount > 0) {
    return activeInstallCount > 0 ? <InstallProgress count={activeInstallCount} /> : null;
  }

  return (
    <Stack {...PANEL_STYLE}>
      <Flex align="center" gap="2">
        <Icon as={BoxIcon} boxSize="4" color="fg.muted" />
        <Text fontSize="sm" fontWeight="700">
          {t('models.launchpad.noModelsTitle')}
        </Text>
      </Flex>
      <Text color="fg.muted" fontSize="xs">
        {t('models.launchpad.noModelsDescription')}
      </Text>
      {activeInstallCount > 0 ? <InstallProgress count={activeInstallCount} /> : <StarterBundles bundles={bundles} />}
    </Stack>
  );
};

const InstallProgress = ({ count }: { count: number }) => {
  const { t } = useTranslation();

  return (
    <Flex align="center" gap="2">
      <Spinner color="fg.muted" size="xs" />
      <Text color="fg.muted" fontSize="xs">
        {t('models.launchpad.installing', { count })}
      </Text>
    </Flex>
  );
};

const StarterBundles = ({ bundles }: { bundles: Record<string, StarterModelBundle> }) => {
  const { t } = useTranslation();
  const entries = useMemo(() => Object.entries(bundles), [bundles]);

  if (entries.length === 0) {
    return null;
  }

  return (
    <Stack gap="2">
      <Text color="fg.muted" fontSize="2xs" fontWeight="600" textTransform="uppercase">
        {t('models.launchpad.starterBundles')}
      </Text>
      <Wrap gap="2">
        {entries.map(([key, bundle]) => (
          <StarterBundleButton bundle={bundle} bundleKey={key} key={key} />
        ))}
      </Wrap>
    </Stack>
  );
};

const StarterBundleButton = ({ bundle, bundleKey }: { bundle: StarterModelBundle; bundleKey: string }) => {
  const navigate = useNavigate();

  const handleOpen = useCallback(() => {
    openAddModelsWithBundle(bundle.name || bundleKey);
    void navigate({ to: '/models' });
  }, [bundle.name, bundleKey, navigate]);

  return (
    <Button size="xs" variant="outline" onClick={handleOpen}>
      {bundle.name || bundleKey}
      <Icon as={ArrowRightIcon} boxSize="3" color="fg.muted" />
    </Button>
  );
};
