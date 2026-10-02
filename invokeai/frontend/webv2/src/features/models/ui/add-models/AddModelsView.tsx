/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelRecordChanges, StarterModel } from '@features/models/core/types';
import type { ElementType } from 'react';

import { Box, Flex, HStack, Icon, Input, InputGroup, Spinner, Stack, Text } from '@chakra-ui/react';
import { collectBases, collectTypes } from '@features/models/core/library';
import {
  DEFAULT_STARTER_MODEL_FILTERS,
  filterStarterModels,
  type StarterModelFilters,
} from '@features/models/core/starters';
import { getHuggingFaceModels, scanFolderForModels, type InstallModelRequest } from '@features/models/data/api';
import {
  ensureExternalProvidersLoaded,
  useExternalProvidersSelector,
} from '@features/models/data/externalProvidersStore';
import { ensureStartersLoaded, useStartersSelector } from '@features/models/data/startersStore';
import {
  clearAddModelsSeeds,
  getAddModelsSeed,
  getAddModelsTypeSeed,
  openExternalProviderKeys,
  openModelManagerTab,
  updateModelsUi,
  useModelsUiSelector,
} from '@features/models/ui/uiStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useScopedAction } from '@platform/react/useScopedAction';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, Scrollable, Tooltip } from '@platform/ui';
import { HuggingFaceIcon } from '@platform/ui/VendoredIcon';
import { DownloadIcon, FileIcon, FolderIcon, FolderSearchIcon, LinkIcon, SearchIcon } from 'lucide-react';
import { useDeferredValue, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { AccessTokenPopover } from './AccessTokenPopover';
import { BundleChips } from './BundleChips';
import { HuggingFaceFiles } from './HuggingFaceFiles';
import { InstallOptions } from './InstallOptions';
import { ScanResults } from './ScanResults';
import { SelectedBundleBar } from './SelectedBundleBar';
import { classifySource } from './sourceClassifier';
import { StarterFilterMenu } from './StarterFilterMenu';
import { StarterList } from './StarterList';
import { getStarterBundleInstallSources, getStarterModelInstallSources } from './starterModelInstallSources';
import { useInstallActions } from './useInstallActions';

/** Keyed by the classifier's label so icon and label can never disagree. */
const SOURCE_KIND_ICONS: Record<string, ElementType> = {
  'models.sourceKind.filePath': FileIcon,
  'models.sourceKind.folderPath': FolderIcon,
  'models.sourceKind.hfRepo': HuggingFaceIcon,
  'models.sourceKind.url': LinkIcon,
};

/**
 * Sent only when ticked. Identification turns FP8 storage on for checkpoints that already store FP8 weights, and an
 * explicit `false` would override that.
 */
const FP8_STORAGE_INSTALL_CONFIG: ModelRecordChanges = { default_settings: { fp8_storage: true } };

export const AddModelsView = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { install, installMany, pendingSources } = useInstallActions();
  const loadError = useStartersSelector((snapshot) => snapshot.error);
  const response = useStartersSelector((snapshot) => snapshot.response);
  const status = useStartersSelector((snapshot) => snapshot.status);
  const { hfLookup, scan, selectedBundleName } = useModelsUiSelector(
    (snapshot) => ({
      hfLookup: snapshot.hfLookup,
      scan: snapshot.scan,
      selectedBundleName: snapshot.selectedBundleName,
    }),
    (left, right) =>
      left.hfLookup === right.hfLookup &&
      left.scan === right.scan &&
      left.selectedBundleName === right.selectedBundleName
  );

  // Read the one-shot seed purely for StrictMode initialization; consume it after mount and keep subsequent input
  // state local.
  const [query, setQuery] = useState(getAddModelsSeed);
  const [accessToken, setAccessToken] = useState('');
  const [inplace, setInplace] = useState(true);
  const [fp8Storage, setFp8Storage] = useState(false);
  const { isBusy: isPulling, run: runPull } = useScopedAction();
  const { isBusy: isScanning, run: runScan } = useScopedAction();
  const scanAbortRef = useRef<AbortController | null>(null);
  const [installingBundle, setInstallingBundle] = useState<string | null>(null);
  const providerConfigs = useExternalProvidersSelector((snapshot) => snapshot.configs);
  const configuredExternalProviders = useMemo<ReadonlySet<string>>(
    () =>
      new Set(
        (providerConfigs ?? []).filter((config) => config.api_key_configured).map((config) => config.provider_id)
      ),
    [providerConfigs]
  );
  const [starterFilters, setStarterFilters] = useState<StarterModelFilters>(() => {
    const typeSeed = getAddModelsTypeSeed();

    return typeSeed === null
      ? DEFAULT_STARTER_MODEL_FILTERS
      : { ...DEFAULT_STARTER_MODEL_FILTERS, typeFilter: typeSeed };
  });

  useMountEffect(() => {
    // One-shot: a seed only ever fills the view it was opened with.
    clearAddModelsSeeds();
    ensureStartersLoaded();

    const owner = captureAccountScope();
    let isMounted = true;

    ensureExternalProvidersLoaded().catch((error: unknown) => {
      if (isMounted && isAccountScopeCurrent(owner)) {
        notify.error(t('models.externalProviderKeysUnavailable'), getApiErrorMessage(error, t('common.unknownError')));
      }
    });

    return () => {
      isMounted = false;
    };
  });

  const trimmed = query.trim();
  const deferredTrimmed = useDeferredValue(trimmed);
  const hasResults = hfLookup !== null || scan !== null;
  const kind = useMemo(() => classifySource(trimmed), [trimmed]);
  const searchIcon = (kind.labelKey ? SOURCE_KIND_ICONS[kind.labelKey] : undefined) ?? SearchIcon;
  const token = accessToken.trim() === '' ? undefined : accessToken.trim();
  const installConfig = fp8Storage ? FP8_STORAGE_INSTALL_CONFIG : undefined;
  // Folders scan; files, URLs, and repos install. Access tokens apply only to URLs.
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

  const filteredModels = useMemo(
    () => filterStarterModels(sourceModels, starterFilters, deferredTrimmed),
    [deferredTrimmed, sourceModels, starterFilters]
  );

  const installStarter = async (model: StarterModel) => {
    const owner = captureAccountScope();
    const queued = await installMany(
      getStarterModelInstallSources(model, { dependencySourcesToSkip: selectedBundleSources })
    );

    if (queued > 0 && isAccountScopeCurrent(owner)) {
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

    const owner = captureAccountScope();
    const bundle = selectedBundle;

    setInstallingBundle(bundle.name);

    try {
      const queued = await installMany(getStarterBundleInstallSources(bundle));

      if (queued > 0 && isAccountScopeCurrent(owner)) {
        notify.success(
          t('models.bundleInstallQueued'),
          t('models.bundleInstallQueuedDescription', { count: queued, name: bundle.name })
        );
      }
    } finally {
      if (isAccountScopeCurrent(owner)) {
        setInstallingBundle(null);
      }
    }
  };

  // Bulk installs queue silently and emit one summary notice rather than per-file toasts.
  const installAllSources = async (requests: InstallModelRequest[]) => {
    const owner = captureAccountScope();
    const queued = await installMany(requests);

    if (queued > 0 && isAccountScopeCurrent(owner)) {
      notify.success(t('models.installAllQueued'), t('models.installAllQueuedDescription', { count: queued }));
    }
  };

  const handlePull = async () => {
    if (!kind.isInstallable) {
      return;
    }

    await runPull(
      async (owner) => {
        if (kind.looksRepo) {
          const lookup = await getHuggingFaceModels(trimmed, owner.signal);

          assertAccountScopeCurrent(owner);
          if (lookup.is_diffusers) {
            updateModelsUi({ hfLookup: null });

            if (
              (await install({ accessToken: token, config: installConfig, source: trimmed })) &&
              isAccountScopeCurrent(owner)
            ) {
              setQuery('');
            }

            return;
          }

          if (!lookup.urls || lookup.urls.length === 0) {
            notify.error(t('models.noModelFilesFoundTitle'), t('models.noInstallableModelFiles'));

            return;
          }

          // Install single-file repositories directly because no selection is needed.
          const [onlyUrl] = lookup.urls;

          if (lookup.urls.length === 1 && onlyUrl) {
            updateModelsUi({ hfLookup: null });

            if (
              (await install({ accessToken: token, config: installConfig, source: onlyUrl })) &&
              isAccountScopeCurrent(owner)
            ) {
              setQuery('');
            }

            return;
          }

          updateModelsUi({ hfLookup: { repo: trimmed, urls: lookup.urls } });

          return;
        }

        if (
          (await install({
            accessToken: token,
            config: installConfig,
            inplace: kind.looksLocal ? inplace : undefined,
            source: trimmed,
          })) &&
          isAccountScopeCurrent(owner)
        ) {
          setQuery('');
        }
      },
      (message) => notify.error(t('models.installFailed'), message)
    );
  };

  // Stop aborts the request; the server watches for the disconnect and
  // abandons its directory walk, so a wrong folder does not keep crawling.
  const handleScan = async () => {
    // Enter in the field reaches here while the button reads Stop; a second
    // scan must not replace the controller the running one is wired to.
    if (scanAbortRef.current) {
      return;
    }

    await runScan(
      async (owner) => {
        const abort = new AbortController();

        scanAbortRef.current = abort;
        try {
          const results = await scanFolderForModels(trimmed, AbortSignal.any([owner.signal, abort.signal]));

          assertAccountScopeCurrent(owner);
          updateModelsUi({ scan: { path: trimmed, results } });
        } finally {
          scanAbortRef.current = null;
        }
      },
      (message, error) => {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          notify.error(t('models.scanFailed'), message);
        }
      }
    );
  };
  const handleStopScan = () => scanAbortRef.current?.abort();

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
              onManageKeys={() => openModelManagerTab('keys')}
            />
          ) : null}

          {canScan && isScanning ? (
            <Button size="sm" variant="outline" onClick={handleStopScan}>
              <Spinner size="xs" />
              {t('models.stopScan')}
            </Button>
          ) : canScan ? (
            <Tooltip content={t('models.scanFolderTooltip')}>
              <Button size="sm" variant="solid" onClick={() => void handleScan()}>
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
                {t('models.toInstallFrom', { source: t(kind.labelKey) })}
              </Text>
            )}
            {canPull ? (
              <InstallOptions
                fp8Storage={fp8Storage}
                inplace={kind.localKind === 'file' ? inplace : undefined}
                onSetFp8Storage={setFp8Storage}
                onSetInplace={setInplace}
              />
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
              onSelect={(name) => updateModelsUi({ selectedBundleName: name })}
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
                fp8Storage={fp8Storage}
                lookup={hfLookup}
                pendingSources={pendingSources}
                onClear={() => updateModelsUi({ hfLookup: null })}
                onInstall={(url) => void install({ accessToken: token, config: installConfig, source: url })}
                onInstallAll={(urls) =>
                  void installAllSources(
                    urls.map((url) => ({ accessToken: token, config: installConfig, source: url }))
                  )
                }
                onSetFp8Storage={setFp8Storage}
              />
            ) : null}

            {scan ? (
              <ScanResults
                fp8Storage={fp8Storage}
                inplace={inplace}
                pendingSources={pendingSources}
                scan={scan}
                onClear={() => updateModelsUi({ scan: null })}
                onInstall={(path) => void install({ config: installConfig, inplace, source: path })}
                onInstallAll={(paths) =>
                  void installAllSources(paths.map((path) => ({ config: installConfig, inplace, source: path })))
                }
                onSetFp8Storage={setFp8Storage}
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
                onConfigureExternalProvider={openExternalProviderKeys}
                onInstall={(model) => void installStarter(model)}
              />
            ) : null}
          </Stack>
        </Scrollable>
      </Box>
    </Flex>
  );
};
