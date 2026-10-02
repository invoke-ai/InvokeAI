import type { ModelConfig, StarterModel } from '@features/models/core/types';
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { StartersSnapshot } from '@features/models/data/startersStore';

import { Badge, Flex, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { getModelBaseColorPalette, getModelBaseLabel } from '@features/models/core/baseIdentity';
import { getModelTypeLabel } from '@features/models/core/taxonomy';
import { useModelsSelector } from '@features/models/data/modelsStore';
import { InstallSourceButton } from '@features/models/ui/shared/InstallSourceButton';
import { findInstalledStarterModelKey, useInstalledSourceKeys } from '@features/models/ui/shared/useInstalledSources';
import { Button } from '@platform/ui';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { KeyRoundIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const getExternalProviderId = (source: string): string | null => {
  if (!source.startsWith('external://')) {
    return null;
  }

  return source.slice('external://'.length).split('/', 1)[0] || null;
};

const selectModels = (snapshot: { models: readonly ModelConfig[] }): readonly ModelConfig[] => snapshot.models;

export const StarterList = ({
  configuredExternalProviders,
  isInstallable,
  loadError,
  models,
  onConfigureExternalProvider,
  onInstall,
  pendingSources,
  response,
  selectedBundleSources,
  status,
}: {
  configuredExternalProviders: ReadonlySet<string>;
  isInstallable: boolean;
  loadError: string | null;
  models: StarterModel[];
  onConfigureExternalProvider: (providerId: string) => void;
  onInstall: (model: StarterModel) => void;
  pendingSources: ReadonlySet<string>;
  response: StartersSnapshot['response'];
  selectedBundleSources: ReadonlySet<string> | undefined;
  status: StartersSnapshot['status'];
}) => {
  const { t } = useTranslation();
  const installedSourceKeys = useInstalledSourceKeys();
  const installedModels = useModelsSelector(selectModels);

  if (status === 'error' && loadError) {
    return (
      <Stack align="center" gap="1" py="8">
        <Text color="fg.error" fontSize="xs" fontWeight="600">
          {t('models.couldNotLoadStarterModels')}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {loadError}
        </Text>
      </Stack>
    );
  }

  if (!response) {
    return (
      <Flex align="center" justify="center" py="10">
        <Spinner color="fg.subtle" size="sm" />
      </Flex>
    );
  }

  if (models.length === 0) {
    return (
      <Text color="fg.muted" fontSize="2xs" py="6" textAlign="center">
        {isInstallable ? t('models.noStarterModelsPull') : t('models.noStarterModelsSearch')}
      </Text>
    );
  }

  return (
    <ListStack dividers label={t('models.starterModelsList')}>
      {models.map((model) => {
        const externalProviderId = getExternalProviderId(model.source);
        const dependencyCount = (model.dependencies ?? []).filter(
          (dependency) => !dependency.is_installed && !selectedBundleSources?.has(dependency.source)
        ).length;
        const trailing = externalProviderId ? (
          configuredExternalProviders.has(externalProviderId) ? (
            <Badge colorPalette="green" fontSize="2xs" size="sm" variant="surface">
              {t('models.installed')}
            </Badge>
          ) : (
            <Button size="2xs" variant="outline" onClick={() => onConfigureExternalProvider(externalProviderId)}>
              <Icon as={KeyRoundIcon} boxSize="3" />
              {t('common.configure')}
            </Button>
          )
        ) : (
          <InstallSourceButton
            installedModelKey={findInstalledStarterModelKey(model, installedSourceKeys, installedModels)}
            isInstalled={model.is_installed}
            isPending={pendingSources.has(model.source)}
            name={model.name}
            source={model.source}
            onInstall={() => onInstall(model)}
          />
        );

        return (
          <ListItem
            key={`${model.source}-${model.name}`}
            actions={trailing}
            badges={
              <>
                <Badge colorPalette={getModelBaseColorPalette(model.base)} fontSize="2xs" size="sm" variant="surface">
                  {getModelBaseLabel(model.base)}
                </Badge>
                <Badge colorPalette="gray" fontSize="2xs" size="sm" variant="surface">
                  {getModelTypeLabel(model.type)}
                </Badge>
              </>
            }
            // Starter descriptions carry the dependency note; allow a second line rather than cutting it off.
            description={
              <Text as="span" lineClamp={2}>
                {`${model.description}${
                  dependencyCount > 0 ? t('models.installsDependencies', { count: dependencyCount }) : ''
                }`}
              </Text>
            }
            title={model.name}
          />
        );
      })}
    </ListStack>
  );
};
