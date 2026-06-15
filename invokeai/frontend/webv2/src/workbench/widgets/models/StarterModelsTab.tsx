import type { ModelTaxonomyType, StarterModel, StarterModelBundle } from '@workbench/models/types';

import {
  Badge,
  Box,
  Flex,
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
import { IconButton, MenuContent, Row, Scrollable, Tooltip } from '@workbench/components/ui';
import { collectBases, collectTypes } from '@workbench/models/library';
import { ensureStartersLoaded, useStartersSnapshot } from '@workbench/models/startersStore';
import { getModelBaseColorPalette, getModelBaseLabel, getModelTypeLabel } from '@workbench/models/taxonomy';
import { useNotify } from '@workbench/useNotify';
import { CheckIcon, DownloadIcon, PackageIcon, SearchIcon, SlidersHorizontalIcon, StarIcon } from 'lucide-react';
import { useEffect, useMemo, useState, type ReactNode } from 'react';

import { FilterMenuItem } from './ModelFilterBar';
import { InstallSourceButton, SourceListItem } from './SourceListItem';
import { useInstallActions } from './useInstallActions';

/**
 * Curated starter models, served from the cached starters store (no refetch
 * on revisit; installs revalidate in background). Bundles sit in a sidebar
 * like playlists: clicking one filters the list to its models, and each has
 * a download action that queues every missing model in the pack.
 */
export const StarterModelsTab = () => {
  const notify = useNotify();
  const { install, pendingSources } = useInstallActions();
  const { error: loadError, response, status } = useStartersSnapshot();
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<ModelTaxonomyType | null>(null);
  const [baseFilter, setBaseFilter] = useState<string | null>(null);
  const [installingBundle, setInstallingBundle] = useState<string | null>(null);
  const [selectedBundleName, setSelectedBundleName] = useState<string | null>(null);

  useEffect(() => {
    ensureStartersLoaded();
  }, []);

  const bundles = useMemo(() => (response ? Object.values(response.starter_bundles) : []), [response]);
  // Falls back to the full catalog if the selected bundle disappears on refresh.
  const selectedBundle = useMemo(
    () => bundles.find((bundle) => bundle.name === selectedBundleName) ?? null,
    [bundles, selectedBundleName]
  );
  const sourceModels = useMemo(
    () => selectedBundle?.models ?? response?.starter_models ?? [],
    [response, selectedBundle]
  );

  const availableTypes = useMemo(() => collectTypes(sourceModels), [sourceModels]);
  const availableBases = useMemo(() => collectBases(sourceModels), [sourceModels]);

  const filteredModels = useMemo(() => {
    const terms = searchTerm.trim().toLowerCase().split(/\s+/).filter(Boolean);

    return sourceModels.filter((model) => {
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
  }, [baseFilter, searchTerm, sourceModels, typeFilter]);

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

  return (
    <HStack align="stretch" gap="4" h="full" minH="0">
      {bundles.length > 0 ? (
        <Stack flexShrink={0} gap="2" minH="0" w="16rem">
          <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
            Bundles
          </Text>
          <Scrollable flex="1" label="Starter bundles" minH="0">
            <Stack gap="1">
              <SidebarRow
                icon={StarIcon}
                isSelected={selectedBundle === null}
                subtitle={`${response.starter_models.length} models`}
                title="All starter models"
                onSelect={() => setSelectedBundleName(null)}
              />
              {bundles.map((bundle) => (
                <BundleRow
                  key={bundle.name}
                  bundle={bundle}
                  isInstalling={installingBundle === bundle.name}
                  isSelected={selectedBundle?.name === bundle.name}
                  onInstall={() => void installBundle(bundle)}
                  onSelect={() => setSelectedBundleName(bundle.name)}
                />
              ))}
            </Stack>
          </Scrollable>
        </Stack>
      ) : null}

      <Stack flex="1" gap="2" minH="0" minW="0">
        <HStack justify="space-between" wrap="wrap">
          <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase" truncate>
            {selectedBundle?.name ?? 'Starter Models'}
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
                  color={typeFilter !== null || baseFilter !== null ? 'accent.solid' : 'fg.muted'}
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
            {selectedBundle
              ? 'No models in this bundle match your search or filters.'
              : 'No starter models match your search or filters.'}
          </Text>
        ) : (
          <Scrollable flex="1" label="Starter models" minH="0" pr="1">
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
          </Scrollable>
        )}
      </Stack>
    </HStack>
  );
};

/** Playlist-style sidebar row: icon, two-line label, optional trailing action. */
const SidebarRow = ({
  icon,
  isSelected,
  onSelect,
  subtitle,
  title,
  trailing,
}: {
  icon: typeof StarIcon;
  isSelected: boolean;
  onSelect: () => void;
  subtitle: string;
  title: string;
  trailing?: ReactNode;
}) => (
  <Row
    active={isSelected ? 'accent' : 'none'}
    aria-current={isSelected || undefined}
    alignItems="start"
    minW="0"
    p="2"
    role="button"
    rounded="md"
    tabIndex={0}
    _focusVisible={{ boxShadow: 'inset 0 0 0 2px {colors.accent.solid}', outline: 'none' }}
    onClick={onSelect}
    onKeyDown={(event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSelect();
      }
    }}
  >
    <Icon as={icon} boxSize="3.5" flexShrink={0} />
    <Stack flex="1" gap="0" minW="0">
      <Text fontSize="xs" fontWeight="600" truncate>
        {title}
      </Text>
      <Text
        color={isSelected ? 'accent.contrast' : 'fg.subtle'}
        opacity={isSelected ? 0.72 : undefined}
        fontSize="2xs"
        truncate
      >
        {subtitle}
      </Text>
    </Stack>
    {trailing}
  </Row>
);

/** A bundle in the sidebar: select to browse its models, download to queue the pack. */
const BundleRow = ({
  bundle,
  isInstalling,
  isSelected,
  onInstall,
  onSelect,
}: {
  bundle: StarterModelBundle;
  isInstalling: boolean;
  isSelected: boolean;
  onInstall: () => void;
  onSelect: () => void;
}) => {
  const missingCount = bundle.models.filter((model) => !model.is_installed).length;

  return (
    <SidebarRow
      icon={PackageIcon}
      isSelected={isSelected}
      subtitle={`${bundle.models.length} models · ${missingCount === 0 ? 'all installed' : `${missingCount} to install`}`}
      title={bundle.name}
      trailing={
        missingCount === 0 ? (
          <Tooltip content="All models in this bundle are installed">
            <Icon as={CheckIcon} boxSize="3.5" color={isSelected ? 'accent.contrast' : 'fg.success'} flexShrink={0} />
          </Tooltip>
        ) : (
          <Tooltip content={`Install ${missingCount} model${missingCount === 1 ? '' : 's'}`}>
            <IconButton
              aria-label={`Install bundle ${bundle.name}`}
              flexShrink={0}
              loading={isInstalling}
              size="2xs"
              variant="ghost"
              color={isSelected ? 'accent.contrast' : undefined}
              // On the accent row, rest at the row's own solid color so the hover
              // transition has an opaque endpoint — animating background to
              // `transparent` interpolates through black and flashes dark on leave.
              bg={isSelected ? 'accent.solid' : undefined}
              _hover={isSelected ? { bg: 'accent.contrast/20' } : undefined}
              onClick={(event) => {
                event.stopPropagation();
                onInstall();
              }}
            >
              <Icon as={DownloadIcon} boxSize="3.5" />
            </IconButton>
          </Tooltip>
        )
      }
      onSelect={onSelect}
    />
  );
};
