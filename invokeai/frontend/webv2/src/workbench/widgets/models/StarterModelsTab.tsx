import {
  Badge,
  Box,
  Flex,
  Grid,
  HoverCard,
  HStack,
  Icon,
  Input,
  InputGroup,
  Menu,
  Portal,
  Spinner,
  Stack,
  Text,
} from '@chakra-ui/react';
import { CheckIcon, DownloadIcon, PackageIcon, SearchIcon, SlidersHorizontalIcon } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { Button, IconButton } from '../../components/ui/Button';
import { MenuContent } from '../../components/ui/Menu';
import { collectBases, collectTypes } from '../../models/library';
import { ensureStartersLoaded, useStartersSnapshot } from '../../models/startersStore';
import { getModelBaseColorPalette, getModelBaseLabel, getModelTypeLabel } from '../../models/taxonomy';
import type { ModelTaxonomyType, StarterModel, StarterModelBundle } from '../../models/types';
import { useNotify } from '../../useNotify';
import { FilterMenuItem } from './ModelFilterBar';
import { InstallSourceButton, SourceListItem } from './SourceListItem';
import { useInstallActions } from './useInstallActions';

/**
 * Curated starter models and one-click bundles, served from the cached
 * starters store (no refetch on revisit; installs revalidate in background).
 * Bundles queue every missing model and dependency in the bundle.
 */
export const StarterModelsTab = () => {
  const notify = useNotify();
  const { install, pendingSources } = useInstallActions();
  const { error: loadError, response, status } = useStartersSnapshot();
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<ModelTaxonomyType | null>(null);
  const [baseFilter, setBaseFilter] = useState<string | null>(null);
  const [installingBundle, setInstallingBundle] = useState<string | null>(null);

  useEffect(() => {
    ensureStartersLoaded();
  }, []);

  const availableTypes = useMemo(() => collectTypes(response?.starter_models ?? []), [response]);
  const availableBases = useMemo(() => collectBases(response?.starter_models ?? []), [response]);

  const filteredModels = useMemo(() => {
    if (!response) {
      return [];
    }

    const terms = searchTerm.trim().toLowerCase().split(/\s+/).filter(Boolean);

    return response.starter_models.filter((model) => {
      if (typeFilter !== null && model.type !== typeFilter) {
        return false;
      }

      if (baseFilter !== null && String(model.base) !== baseFilter) {
        return false;
      }

      if (terms.length === 0) {
        return true;
      }

      const haystack = `${model.name} ${model.description} ${model.base} ${model.type}`.toLowerCase();

      return terms.every((term) => haystack.includes(term));
    });
  }, [baseFilter, response, searchTerm, typeFilter]);

  /** Queue a starter model plus its uninstalled dependencies; returns how many jobs were queued. */
  const queueStarter = async (model: StarterModel): Promise<number> => {
    const sources = [
      ...(model.dependencies ?? []).filter((dependency) => !dependency.is_installed).map((dep) => dep.source),
      model.source,
    ];
    let queued = 0;

    for (const source of sources) {
      if (await install({ source }, { silent: true })) {
        queued += 1;
      }
    }

    return queued;
  };

  const installStarter = async (model: StarterModel) => {
    const queued = await queueStarter(model);

    if (queued > 0) {
      notify.success(
        'Model install queued',
        queued > 1 ? `${model.name} and ${queued - 1} dependenc${queued === 2 ? 'y' : 'ies'}` : model.name
      );
    }
  };

  const installBundle = async (bundle: StarterModelBundle) => {
    setInstallingBundle(bundle.name);

    try {
      let queued = 0;

      for (const model of bundle.models) {
        if (!model.is_installed) {
          queued += await queueStarter(model);
        }
      }

      if (queued > 0) {
        notify.success('Bundle install queued', `${bundle.name}: ${queued} install${queued === 1 ? '' : 's'} queued.`);
      }
    } finally {
      setInstallingBundle(null);
    }
  };

  if (status === 'error' && loadError) {
    return (
      <Stack align="center" gap="1" py="8">
        <Text color="red.400" fontSize="xs" fontWeight="600">
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

  const bundles = Object.values(response.starter_bundles);

  return (
    <Stack gap="4">
      {bundles.length > 0 ? (
        <Stack gap="2">
          <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
            Bundles
          </Text>
          <Grid gap="2" templateColumns="repeat(auto-fill, minmax(15rem, 1fr))">
            {bundles.map((bundle) => (
              <BundleCard
                key={bundle.name}
                bundle={bundle}
                isInstalling={installingBundle === bundle.name}
                onInstall={() => void installBundle(bundle)}
              />
            ))}
          </Grid>
        </Stack>
      ) : null}

      <Stack gap="2">
        <HStack justify="space-between" wrap="wrap">
          <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
            Starter Models
          </Text>
          <HStack gap="1.5">
            <Box maxW="18rem" w="full">
              <InputGroup startElement={<Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />}>
                <Input
                  aria-label="Search starter models"
                  placeholder="Search starter models…"
                  size="sm"
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.currentTarget.value)}
                />
              </InputGroup>
            </Box>
            <Menu.Root closeOnSelect={false} positioning={{ placement: 'bottom-end' }}>
              <Menu.Trigger asChild>
                <IconButton
                  aria-label="Filter starter models"
                  color={typeFilter !== null || baseFilter !== null ? 'accent.invoke' : 'fg.muted'}
                  size="sm"
                  variant="ghost"
                >
                  <Icon as={SlidersHorizontalIcon} boxSize="4" />
                </IconButton>
              </Menu.Trigger>
              <Portal>
                <Menu.Positioner>
                  <MenuContent minW="13rem">
                    <Menu.ItemGroup>
                      <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                        Model Type
                      </Menu.ItemGroupLabel>
                      <FilterMenuItem
                        isChecked={typeFilter === null}
                        label="All types"
                        value="type-all"
                        onSelect={() => setTypeFilter(null)}
                      />
                      {availableTypes.map((type) => (
                        <FilterMenuItem
                          key={type}
                          isChecked={typeFilter === type}
                          label={getModelTypeLabel(type)}
                          value={`type-${type}`}
                          onSelect={() => setTypeFilter((current) => (current === type ? null : type))}
                        />
                      ))}
                    </Menu.ItemGroup>
                    <Menu.Separator />
                    <Menu.ItemGroup>
                      <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                        Base Architecture
                      </Menu.ItemGroupLabel>
                      <FilterMenuItem
                        isChecked={baseFilter === null}
                        label="All bases"
                        value="base-all"
                        onSelect={() => setBaseFilter(null)}
                      />
                      {availableBases.map((base) => (
                        <FilterMenuItem
                          key={base}
                          isChecked={baseFilter === base}
                          label={getModelBaseLabel(base)}
                          value={`base-${base}`}
                          onSelect={() => setBaseFilter((current) => (current === base ? null : base))}
                        />
                      ))}
                    </Menu.ItemGroup>
                  </MenuContent>
                </Menu.Positioner>
              </Portal>
            </Menu.Root>
          </HStack>
        </HStack>
        {filteredModels.length === 0 ? (
          <Text color="fg.subtle" fontSize="2xs" py="4" textAlign="center">
            No starter models match your search or filters.
          </Text>
        ) : (
          <Stack gap="1.5">
            {filteredModels.map((model) => {
              const dependencyCount = model.dependencies?.length ?? 0;

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
                  trailing={
                    <InstallSourceButton
                      isInstalled={model.is_installed}
                      isPending={pendingSources.has(model.source)}
                      source={model.source}
                      onInstall={() => void installStarter(model)}
                    />
                  }
                />
              );
            })}
          </Stack>
        )}
      </Stack>
    </Stack>
  );
};

/**
 * Bundle card whose model list previews in a hover card. The whole card is
 * the hover trigger so the preview also works when the install button is
 * disabled (all models installed).
 */
const BundleCard = ({
  bundle,
  isInstalling,
  onInstall,
}: {
  bundle: StarterModelBundle;
  isInstalling: boolean;
  onInstall: () => void;
}) => {
  const missingCount = bundle.models.filter((model) => !model.is_installed).length;

  return (
    <HoverCard.Root closeDelay={100} openDelay={250} positioning={{ placement: 'bottom' }}>
      <HoverCard.Trigger asChild>
        <Stack bg="bg.surface" borderColor="border.subtle" borderWidth="1px" gap="2" p="3" rounded="lg">
          <HStack gap="2">
            <Icon as={PackageIcon} boxSize="4" color="fg.muted" />
            <Text flex="1" fontSize="xs" fontWeight="700" minW="0" truncate>
              {bundle.name}
            </Text>
          </HStack>
          <Text color="fg.subtle" fontSize="2xs">
            {bundle.models.length} models · {missingCount === 0 ? 'all installed' : `${missingCount} to install`}
          </Text>
          <Button
            disabled={missingCount === 0}
            loading={isInstalling}
            size="xs"
            variant={missingCount === 0 ? 'outline' : 'solid'}
            w="full"
            onClick={onInstall}
          >
            <Icon as={DownloadIcon} boxSize="3" />
            {missingCount === 0 ? 'Installed' : 'Install Bundle'}
          </Button>
        </Stack>
      </HoverCard.Trigger>
      <Portal>
        <HoverCard.Positioner>
          <HoverCard.Content
            bg="bg.surfaceRaised"
            borderColor="border.emphasis"
            borderWidth="1px"
            color="fg.default"
            p="3"
            rounded="lg"
            shadow="lg"
            w="22rem"
          >
            <HoverCard.Arrow>
              <HoverCard.ArrowTip />
            </HoverCard.Arrow>
            {/* Plain overflow box: ScrollArea's content grows to fit long
                unbroken names, which defeats truncation. */}
            <Box aria-label={`Models in ${bundle.name}`} maxH="18rem" overflowY="auto">
              <Stack gap="1.5">
                <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
                  In this bundle
                </Text>
                {bundle.models.map((model) => (
                  <HStack key={`${model.source}-${model.name}`} gap="1.5" minW="0" w="full">
                    <Icon
                      as={model.is_installed ? CheckIcon : DownloadIcon}
                      boxSize="3"
                      color={model.is_installed ? 'green.400' : 'fg.subtle'}
                      flexShrink={0}
                    />
                    <Text flex="1" fontSize="2xs" minW="0" title={model.name} truncate>
                      {model.name}
                    </Text>
                    <Badge
                      colorPalette={getModelBaseColorPalette(model.base)}
                      flexShrink={0}
                      fontSize="2xs"
                      size="sm"
                      variant="surface"
                    >
                      {getModelBaseLabel(model.base)}
                    </Badge>
                  </HStack>
                ))}
              </Stack>
            </Box>
          </HoverCard.Content>
        </HoverCard.Positioner>
      </Portal>
    </HoverCard.Root>
  );
};
