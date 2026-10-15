import type { ModelConfig, StarterModel } from '@features/models/core/types';
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { StartersSnapshot } from '@features/models/data/startersStore';
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';

import { Badge, Flex, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { getModelBaseColorPalette, getModelBaseLabel } from '@features/models/core/baseIdentity';
import { getModelTypeLabel } from '@features/models/core/taxonomy';
import { useModelsSelector } from '@features/models/data/modelsStore';
import {
  InstallSourceContextMenu,
  type InstallSourceStatus,
  useInstallSourceMenu,
} from '@features/models/ui/shared/InstallSourceMenu';
import { InstallSourceRow } from '@features/models/ui/shared/InstallSourceRow';
import { findInstalledStarterModelKey, useInstalledSourceKeys } from '@features/models/ui/shared/useInstalledSources';
import { Button } from '@platform/ui';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { KeyRoundIcon } from 'lucide-react';
import { memo } from 'react';
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
  const statusOf = (model: StarterModel): InstallSourceStatus => ({
    installedModelKey: findInstalledStarterModelKey(model, installedSourceKeys, installedModels),
    isInstalled: model.is_installed,
    isPending: pendingSources.has(model.source),
  });
  const menu = useInstallSourceMenu();
  // External-provider rows have no menu, so only an installable source can be the target.
  const menuModel = menu.shown ? models.find((model) => model.source === menu.shown?.source) : undefined;

  if (status === 'error' && loadError) {
    return (
      <Stack align="center" gap="1" py="8">
        <Text color="fg.error" fontSize="md" fontWeight="600">
          {t('models.couldNotLoadStarterModels')}
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          {loadError}
        </Text>
      </Stack>
    );
  }

  if (!response) {
    return (
      <Flex align="center" justify="center" py="10">
        <Spinner color="fg.subtle" size="lg" />
      </Flex>
    );
  }

  if (models.length === 0) {
    return (
      <Text color="fg.muted" fontSize="xs" py="6" textAlign="center">
        {isInstallable ? t('models.noStarterModelsPull') : t('models.noStarterModelsSearch')}
      </Text>
    );
  }

  return (
    <>
      <ListStack dividers label={t('models.starterModelsList')}>
        {models.map((model) => {
          const externalProviderId = getExternalProviderId(model.source);
          const dependencyCount = (model.dependencies ?? []).filter(
            (dependency) => !dependency.is_installed && !selectedBundleSources?.has(dependency.source)
          ).length;
          const description = `${model.description}${
            dependencyCount > 0 ? t('models.installsDependencies', { count: dependencyCount }) : ''
          }`;

          if (!externalProviderId) {
            return (
              <StarterSourceRow
                key={`${model.source}-${model.name}`}
                {...statusOf(model)}
                description={description}
                isMenuOpen={menuModel === model}
                model={model}
                source={model.source}
                onInstall={onInstall}
                onOpenMenu={menu.open}
              />
            );
          }

          return (
            <ListItem
              key={`${model.source}-${model.name}`}
              actions={
                configuredExternalProviders.has(externalProviderId) ? (
                  <Badge colorPalette="green" fontSize="xs" size="lg" variant="surface">
                    {t('models.installed')}
                  </Badge>
                ) : (
                  <Button size="sm" variant="outline" onClick={() => onConfigureExternalProvider(externalProviderId)}>
                    <Icon as={KeyRoundIcon} boxSize="3" />
                    {t('common.configure')}
                  </Button>
                )
              }
              badges={<StarterBadges model={model} />}
              description={<StarterDescription text={description} />}
              title={model.name}
            />
          );
        })}
      </ListStack>
      <InstallSourceContextMenu
        status={menuModel ? statusOf(menuModel) : null}
        isOpen={menu.target !== null}
        target={menu.shown}
        onClose={menu.close}
        onExitComplete={menu.release}
        onInstall={() => {
          if (menuModel) {
            onInstall(menuModel);
          }
        }}
      />
    </>
  );
};

const StarterBadges = ({ model }: { model: StarterModel }) => (
  <>
    <Badge colorPalette={getModelBaseColorPalette(model.base)} fontSize="xs" size="lg" variant="surface">
      {getModelBaseLabel(model.base)}
    </Badge>
    <Badge colorPalette="gray" fontSize="xs" size="lg" variant="surface">
      {getModelTypeLabel(model.type)}
    </Badge>
  </>
);

/** Starter descriptions carry the dependency note; allow a second line rather than cutting it off. */
const StarterDescription = ({ text }: { text: string }) => (
  <Text as="span" lineClamp={2}>
    {text}
  </Text>
);

/** Memoized so opening the list's menu re-renders only the targeted row; the list passes stable callbacks. */
const StarterSourceRow = memo(function StarterSourceRow({
  description,
  model,
  onInstall,
  ...row
}: InstallSourceStatus & {
  description: string;
  isMenuOpen: boolean;
  model: StarterModel;
  onInstall: (model: StarterModel) => void;
  onOpenMenu: (source: string, anchor: ListContextMenuAnchor) => void;
  source: string;
}) {
  return (
    <InstallSourceRow
      {...row}
      badges={<StarterBadges model={model} />}
      description={<StarterDescription text={description} />}
      name={model.name}
      title={model.name}
      onInstall={() => onInstall(model)}
    />
  );
});
