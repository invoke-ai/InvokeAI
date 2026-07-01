/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelSortField } from '@workbench/models/library';
import type { StarterModel } from '@workbench/models/types';

import { Box, Checkbox, Flex, HStack, Icon, Input, InputGroup, Stack, Text } from '@chakra-ui/react';
import { Button, Scrollable, Tooltip } from '@workbench/components/ui';
import { getExternalProviderConfigs, getHuggingFaceModels, scanFolderForModels } from '@workbench/models/api';
import { getModelBaseLabel } from '@workbench/models/baseIdentity';
import { collectBases, collectTypes } from '@workbench/models/library';
import { ensureStartersLoaded, useStartersSelector } from '@workbench/models/startersStore';
import { updateModelsUi, useModelsUiSelector } from '@workbench/models/uiStore';
import { useNotify } from '@workbench/useNotify';
import { DownloadIcon, FileIcon, FolderIcon, FolderSearchIcon, LinkIcon, SearchIcon } from 'lucide-react';
import { useDeferredValue, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SiHuggingface } from 'react-icons/si';

import type { StarterInstallSource } from './starterModelInstallSources';

import { AccessTokenPopover } from './AccessTokenPopover';
import { BundleChips } from './BundleChips';
import { HuggingFaceFiles } from './HuggingFaceFiles';
import { ScanResults } from './ScanResults';
import { SelectedBundleBar } from './SelectedBundleBar';
import { classifySource } from './sourceClassifier';
import { DEFAULT_STARTER_MODEL_FILTERS, StarterFilterMenu, type StarterModelFilters } from './StarterFilterMenu';
import { StarterList } from './StarterList';
import { getStarterBundleInstallSources, getStarterModelInstallSources } from './starterModelInstallSources';
import { useInstallActions } from './useInstallActions';

interface IndexedStarterModel {
  index: number;
  model: StarterModel;
}

const compareStarterModels = (a: IndexedStarterModel, b: IndexedStarterModel, field: ModelSortField): number => {
  switch (field) {
    case 'default':
      return a.index - b.index;
    case 'name':
      return a.model.name.localeCompare(b.model.name, undefined, { sensitivity: 'base' });
    case 'base':
      return getModelBaseLabel(a.model.base).localeCompare(getModelBaseLabel(b.model.base));
    case 'format':
      return String(a.model.format ?? '').localeCompare(String(b.model.format ?? ''));
    case 'size':
      return 0;
  }
};

/**
 * One box to add any model. The same field searches the curated starter
 * catalog and accepts a URL, local path, or HuggingFace repo to install
 * directly. Source-specific result panels live under `add-models/` so this file
 * stays focused on state and install orchestration.
 */
export const AddModelsView = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { install, pendingSources } = useInstallActions();
  const loadError = useStartersSelector((snapshot) => snapshot.error);
  const response = useStartersSelector((snapshot) => snapshot.response);
  const status = useStartersSelector((snapshot) => snapshot.status);
  const { hfLookup, scan } = useModelsUiSelector(
    (snapshot) => ({ hfLookup: snapshot.hfLookup, scan: snapshot.scan }),
    (left, right) => left.hfLookup === right.hfLookup && left.scan === right.scan
  );

  const [query, setQuery] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const [inplace, setInplace] = useState(true);
  const [selectedBundleName, setSelectedBundleName] = useState<string | null>(null);
  const [isPulling, setIsPulling] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [installingBundle, setInstallingBundle] = useState<string | null>(null);
  const [configuredExternalProviders, setConfiguredExternalProviders] = useState<ReadonlySet<string>>(() => new Set());
  const [starterFilters, setStarterFilters] = useState<StarterModelFilters>(DEFAULT_STARTER_MODEL_FILTERS);

  useEffect(() => {
    ensureStartersLoaded();
  }, []);

  useEffect(() => {
    let isStale = false;

    getExternalProviderConfigs()
      .then((configs) => {
        if (isStale) {
          return;
        }

        setConfiguredExternalProviders(
          new Set(configs.filter((config) => config.api_key_configured).map((config) => config.provider_id))
        );
      })
      .catch((error) => {
        if (isStale) {
          return;
        }

        setConfiguredExternalProviders(new Set());
        notify.error(
          t('models.externalProviderKeysUnavailable'),
          error instanceof Error ? error.message : String(error)
        );
      });

    return () => {
      isStale = true;
    };
  }, [notify, t]);

  const trimmed = query.trim();
  const deferredTrimmed = useDeferredValue(trimmed);
  // A pull/scan results panel is showing: focus on the results and hide the
  // browse-only chrome (bundles, filter menu, starter catalog).
  const hasResults = hfLookup !== null || scan !== null;
  const kind = useMemo(() => classifySource(trimmed), [trimmed]);
  const searchIcon = kind.looksRepo
    ? SiHuggingface
    : kind.looksUrl
      ? LinkIcon
      : kind.localKind === 'file'
        ? FileIcon
        : kind.localKind === 'folder'
          ? FolderIcon
          : SearchIcon;
  const token = accessToken.trim() === '' ? undefined : accessToken.trim();
  // The primary action depends on the detected input: folders are scanned for
  // models; files, URLs, and HF repos are pulled. Access tokens only apply to URLs.
  const canScan = kind.localKind === 'folder';
  const canPull = kind.isInstallable && !canScan;

  const bundles = useMemo(() => (response ? Object.values(response.starter_bundles) : []), [response]);
  const selectedBundle = useMemo(
    () => bundles.find((bundle) => bundle.name === selectedBundleName) ?? null,
    [bundles, selectedBundleName]
  );
  const sourceModels = useMemo(
    () => selectedBundle?.models ?? response?.starter_models ?? [],
    [response, selectedBundle]
  );
  const selectedBundleSources = useMemo(
    () => (selectedBundle ? new Set(selectedBundle.models.map((model) => model.source)) : undefined),
    [selectedBundle]
  );
  const availableStarterBases = useMemo(() => collectBases(sourceModels), [sourceModels]);
  const availableStarterTypes = useMemo(() => collectTypes(sourceModels), [sourceModels]);

  const filteredModels = useMemo(() => {
    const terms = deferredTrimmed.toLowerCase().split(/\s+/).filter(Boolean);
    const directionFactor = starterFilters.sortDirection === 'desc' ? -1 : 1;

    return sourceModels
      .map((model, index) => ({ index, model }))
      .filter(({ model }) => {
        const haystack =
          `${model.name} ${model.description} ${model.base} ${model.type} ${model.format ?? ''} ${model.variant ?? ''}`.toLowerCase();

        return (
          (starterFilters.typeFilter === null || model.type === starterFilters.typeFilter) &&
          (starterFilters.baseFilter === null || model.base === starterFilters.baseFilter) &&
          terms.every((term) => haystack.includes(term))
        );
      })
      .sort((a, b) => compareStarterModels(a, b, starterFilters.sortField) * directionFactor)
      .map(({ model }) => model);
  }, [deferredTrimmed, sourceModels, starterFilters]);

  const queueSources = async (entries: StarterInstallSource[]): Promise<number> => {
    let queued = 0;

    for (const { config, source } of entries) {
      if (await install({ config, source }, { silent: true })) {
        queued += 1;
      }
    }

    return queued;
  };

  const installStarter = async (model: StarterModel) => {
    const queued = await queueSources(
      getStarterModelInstallSources(model, { dependencySourcesToSkip: selectedBundleSources })
    );

    if (queued > 0) {
      notify.success(
        t('models.modelInstallQueued'),
        queued > 1 ? t('models.modelAndDependenciesQueued', { count: queued - 1, name: model.name }) : model.name
      );
    }
  };

  const installBundle = async () => {
    if (!selectedBundle) {
      return;
    }

    setInstallingBundle(selectedBundle.name);

    try {
      const queued = await queueSources(getStarterBundleInstallSources(selectedBundle));

      if (queued > 0) {
        notify.success(
          t('models.bundleInstallQueued'),
          t('models.bundleInstallQueuedDescription', { count: queued, name: selectedBundle.name })
        );
      }
    } finally {
      setInstallingBundle(null);
    }
  };

  const handlePull = async () => {
    if (!kind.isInstallable) {
      return;
    }

    setIsPulling(true);

    try {
      if (kind.looksRepo) {
        const lookup = await getHuggingFaceModels(trimmed);

        if (lookup.is_diffusers) {
          updateModelsUi({ hfLookup: null });

          if (await install({ accessToken: token, source: trimmed })) {
            setQuery('');
          }

          return;
        }

        if (!lookup.urls || lookup.urls.length === 0) {
          notify.error(t('models.noModelFilesFoundTitle'), t('models.noInstallableModelFiles'));

          return;
        }

        // A single-file repo has nothing to choose from — install it directly,
        // mirroring the diffusers path, instead of showing a one-row list.
        const [onlyUrl] = lookup.urls;

        if (lookup.urls.length === 1 && onlyUrl) {
          updateModelsUi({ hfLookup: null });

          if (await install({ accessToken: token, source: onlyUrl })) {
            setQuery('');
          }

          return;
        }

        updateModelsUi({ hfLookup: { repo: trimmed, urls: lookup.urls } });

        return;
      }

      if (await install({ accessToken: token, inplace: kind.looksLocal ? inplace : undefined, source: trimmed })) {
        setQuery('');
      }
    } catch (error) {
      notify.error(t('models.installFailed'), error instanceof Error ? error.message : String(error));
    } finally {
      setIsPulling(false);
    }
  };

  const handleScan = async () => {
    setIsScanning(true);

    try {
      updateModelsUi({ scan: { path: trimmed, results: await scanFolderForModels(trimmed) } });
    } catch (error) {
      notify.error(t('models.scanFailed'), error instanceof Error ? error.message : String(error));
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <Flex direction="column" h="full" minH="0">
      <Stack gap="2" pb="2" pt="3">
        <HStack align="center" gap="2" px="3">
          <InputGroup flex="1" startElement={<Icon as={searchIcon} boxSize="3.5" color="fg.subtle" />}>
            <Input
              aria-label={t('models.searchOrAdd')}
              placeholder={t('models.searchOrAddPlaceholder')}
              size="sm"
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
              onKeyDown={(event) => {
                if (event.key !== 'Enter') {
                  return;
                }

                if (canScan) {
                  event.preventDefault();
                  void handleScan();
                } else if (canPull) {
                  event.preventDefault();
                  void handlePull();
                }
              }}
            />
          </InputGroup>

          {kind.looksUrl ? (
            <AccessTokenPopover
              value={accessToken}
              onChange={setAccessToken}
              onManageKeys={() => updateModelsUi({ activeTab: 'keys' })}
            />
          ) : null}

          {canScan ? (
            <Tooltip content={t('models.scanFolderTooltip')}>
              <Button loading={isScanning} size="sm" variant="solid" onClick={() => void handleScan()}>
                <Icon as={FolderSearchIcon} boxSize="3.5" />
                {t('models.scan')}
              </Button>
            </Tooltip>
          ) : null}

          {canPull ? (
            <Button loading={isPulling} size="sm" variant="solid" onClick={() => void handlePull()}>
              <Icon as={DownloadIcon} boxSize="3.5" />
              {t('models.pull')}
            </Button>
          ) : null}
        </HStack>

        {kind.isInstallable && !hasResults ? (
          <HStack color="fg.subtle" fontSize="2xs" gap="2" px="3" wrap="wrap">
            {canScan ? (
              <Text>
                {t('models.press')}{' '}
                <Text as="span" color="fg.muted" fontWeight="600">
                  {t('models.scan')}
                </Text>{' '}
                {t('models.toFindModelsInFolder')}
              </Text>
            ) : (
              <Text>
                {t('models.press')}{' '}
                <Text as="span" color="fg.muted" fontWeight="600">
                  {t('models.pull')}
                </Text>{' '}
                {t('models.toInstallFrom', { source: kind.label })}
              </Text>
            )}
            {kind.localKind === 'file' ? (
              <Checkbox.Root
                checked={inplace}
                colorPalette="accent"
                size="xs"
                onCheckedChange={(event) => setInplace(event.checked === true)}
              >
                <Checkbox.HiddenInput />
                <Checkbox.Control />
                <Checkbox.Label fontSize="2xs">{t('models.installInPlace')}</Checkbox.Label>
              </Checkbox.Root>
            ) : null}
          </HStack>
        ) : null}

        {!hasResults ? (
          <>
            <BundleChips
              bundles={bundles}
              selectedName={selectedBundleName}
              starterCount={response?.starter_models.length ?? 0}
              trailing={
                response ? (
                  <StarterFilterMenu
                    availableBases={availableStarterBases}
                    availableTypes={availableStarterTypes}
                    filters={starterFilters}
                    onChange={setStarterFilters}
                  />
                ) : undefined
              }
              onSelect={setSelectedBundleName}
            />

            {selectedBundle ? (
              <SelectedBundleBar
                bundle={selectedBundle}
                isInstalling={installingBundle === selectedBundle.name}
                onInstall={() => void installBundle()}
              />
            ) : null}
          </>
        ) : null}
      </Stack>

      <Box flex="1" minH="0">
        <Scrollable h="full" label={t('models.addModelsResults')} minH="0" px="3" pb="3">
          <Stack gap="3">
            {hfLookup ? (
              <HuggingFaceFiles
                lookup={hfLookup}
                pendingSources={pendingSources}
                onClear={() => updateModelsUi({ hfLookup: null })}
                onInstall={(url) => void install({ accessToken: token, source: url })}
              />
            ) : null}

            {scan ? (
              <ScanResults
                inplace={inplace}
                pendingSources={pendingSources}
                scan={scan}
                onClear={() => updateModelsUi({ scan: null })}
                onInstall={(path) => void install({ inplace, source: path })}
                onSetInplace={setInplace}
              />
            ) : null}

            {!hasResults ? (
              <StarterList
                configuredExternalProviders={configuredExternalProviders}
                isInstallable={kind.isInstallable}
                loadError={loadError}
                models={filteredModels}
                pendingSources={pendingSources}
                response={response}
                selectedBundleSources={selectedBundleSources}
                status={status}
                onConfigureExternalProvider={() => updateModelsUi({ activeTab: 'keys' })}
                onInstall={(model) => void installStarter(model)}
              />
            ) : null}
          </Stack>
        </Scrollable>
      </Box>
    </Flex>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
