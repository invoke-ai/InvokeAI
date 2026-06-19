import type { useStartersSnapshot } from '@workbench/models/startersStore';
import type { StarterModel } from '@workbench/models/types';

import { Badge, Flex, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { InstallSourceButton, SourceListItem } from '@workbench/launchpad/models/shared/SourceListItem';
import { getModelBaseColorPalette, getModelBaseLabel } from '@workbench/models/baseIdentity';
import { getModelTypeLabel } from '@workbench/models/taxonomy';
import { KeyRoundIcon } from 'lucide-react';

const getExternalProviderId = (source: string): string | null => {
  if (!source.startsWith('external://')) {
    return null;
  }

  return source.slice('external://'.length).split('/', 1)[0] || null;
};

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
  onConfigureExternalProvider: () => void;
  onInstall: (model: StarterModel) => void;
  pendingSources: ReadonlySet<string>;
  response: ReturnType<typeof useStartersSnapshot>['response'];
  selectedBundleSources: ReadonlySet<string> | undefined;
  status: ReturnType<typeof useStartersSnapshot>['status'];
}) => {
  if (status === 'error' && loadError) {
    return (
      <Stack align="center" gap="1" py="8">
        <Text color="fg.error" fontSize="xs" fontWeight="600">
          Could not load starter models
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
      <Text color="fg.subtle" fontSize="2xs" py="6" textAlign="center">
        {isInstallable
          ? 'No starter models match - press Pull to install the source you entered.'
          : 'No starter models match your search.'}
      </Text>
    );
  }

  return (
    <Stack gap="1.5">
      {models.map((model) => {
        const externalProviderId = getExternalProviderId(model.source);
        const dependencyCount = (model.dependencies ?? []).filter(
          (dependency) => !dependency.is_installed && !selectedBundleSources?.has(dependency.source)
        ).length;
        const trailing = externalProviderId ? (
          configuredExternalProviders.has(externalProviderId) ? (
            <Badge colorPalette="green" flexShrink={0} fontSize="2xs" size="sm" variant="surface">
              Installed
            </Badge>
          ) : (
            <Button
              flexShrink={0}
              size="2xs"
              variant="outline"
              onClick={(event) => {
                event.stopPropagation();
                onConfigureExternalProvider();
              }}
            >
              <Icon as={KeyRoundIcon} boxSize="3" />
              Configure
            </Button>
          )
        ) : (
          <InstallSourceButton
            isInstalled={model.is_installed}
            isPending={pendingSources.has(model.source)}
            source={model.source}
            onInstall={() => onInstall(model)}
          />
        );

        return (
          <SourceListItem
            key={`${model.source}-${model.name}`}
            badges={
              <>
                <Badge
                  colorPalette={getModelBaseColorPalette(model.base)}
                  flexShrink={0}
                  fontSize="2xs"
                  size="sm"
                  variant="surface"
                >
                  {getModelBaseLabel(model.base)}
                </Badge>
                <Badge colorPalette="gray" flexShrink={0} fontSize="2xs" size="sm" variant="outline">
                  {getModelTypeLabel(model.type)}
                </Badge>
              </>
            }
            description={`${model.description}${
              dependencyCount > 0
                ? ` (installs ${dependencyCount} dependenc${dependencyCount === 1 ? 'y' : 'ies'})`
                : ''
            }`}
            title={model.name}
            trailing={trailing}
          />
        );
      })}
    </Stack>
  );
};
