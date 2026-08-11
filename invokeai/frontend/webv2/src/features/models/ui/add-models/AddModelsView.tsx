/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { StarterModel } from '@features/models/core/types';

import { Box, Checkbox, Flex, HStack, Icon, Input, InputGroup, Stack, Text } from '@chakra-ui/react';
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
import { updateModelsUi, useModelsUiSelector } from '@features/models/ui/uiStore';
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
import { DownloadIcon, FileIcon, FolderIcon, FolderSearchIcon, LinkIcon, SearchIcon } from 'lucide-react';
import { useDeferredValue, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SiHuggingface } from 'react-icons/si';

import { AccessTokenPopover } from './AccessTokenPopover';
import { BundleChips } from './BundleChips';
import { HuggingFaceFiles } from './HuggingFaceFiles';
import { ScanResults } from './ScanResults';
import { SelectedBundleBar } from './SelectedBundleBar';
import { classifySource } from './sourceClassifier';
import { StarterFilterMenu } from './StarterFilterMenu';
import { StarterList } from './StarterList';
import { getStarterBundleInstallSources, getStarterModelInstallSources } from './starterModelInstallSources';
import { useInstallActions } from './useInstallActions';

/**
 * One box to add any model. The same field searches the curated starter
 * catalog and accepts a URL, local path, or HuggingFace repo to install
 * directly. Source-specific result panels live under `add-models/` so this file
 * stays focused on state and install orchestration.
 */
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

  const [query, setQuery] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const [inplace, setInplace] = useState(true);
  const { isBusy: isPulling, run: runPull } = useScopedAction();
  const { isBusy: isScanning, run: runScan } = useScopedAction();
  const [installingBundle, setInstallingBundle] = useState<string | null>(null);
  const providerConfigs = useExternalProvidersSelector((snapshot) => snapshot.configs);
  const configuredExternalProviders = useMemo<ReadonlySet<string>>(
    () =>
      new Set(
        (providerConfigs ?? []).filter((config) => config.api_key_configured).map((config) => config.provider_id)
      ),
    [providerConfigs]
  );
  const [starterFilters, setStarterFilters] = useState<StarterModelFilters>(DEFAULT_STARTER_MODEL_FILTERS);

  useMountEffect(() => {
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

  // Install-all from a results panel: silent per-model queueing with one
  // summary toast, the same shape the bundle path uses — never a toast per
  // file for a 40-file repo.
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

            if ((await install({ accessToken: token, source: trimmed })) && isAccountScopeCurrent(owner)) {
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

            if ((await install({ accessToken: token, source: onlyUrl })) && isAccountScopeCurrent(owner)) {
              setQuery('');
            }

            return;
          }

          updateModelsUi({ hfLookup: { repo: trimmed, urls: lookup.urls } });

          return;
        }

        if (
          (await install({ accessToken: token, inplace: kind.looksLocal ? inplace : undefined, source: trimmed })) &&
          isAccountScopeCurrent(owner)
        ) {
          setQuery('');
        }
      },
      (message) => notify.error(t('models.installFailed'), message)
    );
  };

  const handleScan = async () => {
    await runScan(
      async (owner) => {
        const results = await scanFolderForModels(trimmed, owner.signal);

        assertAccountScopeCurrent(owner);
        updateModelsUi({ scan: { path: trimmed, results } });
      },
      (message) => notify.error(t('models.scanFailed'), message)
    );
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
                {t('models.toInstallFrom', { source: kind.labelKey ? t(kind.labelKey) : '' })}
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
                lookup={hfLookup}
                pendingSources={pendingSources}
                onClear={() => updateModelsUi({ hfLookup: null })}
                onInstall={(url) => void install({ accessToken: token, source: url })}
                onInstallAll={(urls) =>
                  void installAllSources(urls.map((url) => ({ accessToken: token, source: url })))
                }
              />
            ) : null}

            {scan ? (
              <ScanResults
                inplace={inplace}
                pendingSources={pendingSources}
                scan={scan}
                onClear={() => updateModelsUi({ scan: null })}
                onInstall={(path) => void install({ inplace, source: path })}
                onInstallAll={(paths) => void installAllSources(paths.map((path) => ({ inplace, source: path })))}
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
