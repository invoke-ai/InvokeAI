import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Badge,
  Button,
  Card,
  Flex,
  FormControl,
  FormHelperText,
  FormLabel,
  Heading,
  Input,
  Switch,
  Text,
  Tooltip,
  useToast,
} from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useBuildModelInstallArg } from 'features/modelManagerV2/hooks/useBuildModelsToInstall';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { IconType } from 'react-icons';
import { PiCheckBold, PiCloudLightningBold, PiWarningBold } from 'react-icons/pi';
import { SiAlibabacloud, SiBytedance, SiGooglegemini, SiOpenai } from 'react-icons/si';
import {
  useGetExternalProviderConfigsQuery,
  useInstallFalModelMutation,
  useLazyGetFalModelsQuery,
  useResetExternalProviderConfigMutation,
  useSetExternalProviderConfigMutation,
} from 'services/api/endpoints/appInfo';
import { useGetStarterModelsQuery } from 'services/api/endpoints/models';
import type { ExternalProviderConfig, FalCatalogModel, StarterModel } from 'services/api/types';

import { isFalNativeCanvasModel } from './falCatalog';

const PROVIDER_SORT_ORDER = ['fal', 'gemini', 'openai', 'seedream', 'alibabacloud'];

function resolveProviderIcon(providerId: string): IconType | null {
  const provider = providerId.toLowerCase();

  switch (provider) {
    case 'fal':
      return PiCloudLightningBold;
    case 'openai':
      return SiOpenai;
    case 'gemini':
      return SiGooglegemini;
    case 'seedream':
      return SiBytedance;
    case 'alibabacloud':
      return SiAlibabacloud;
    default:
      return null;
  }
}

const FORM_CONTROL_SX: SystemStyleObject = {
  flexDir: 'column',
  alignItems: 'flex-start',
  gap: 2,
};

type ProviderCardProps = {
  provider: ExternalProviderConfig;
  onInstallModels: (providerId: string) => void;
  iconResolver: (providerId: string) => IconType | null;
};

type UpdatePayload = {
  provider_id: string;
  api_key?: string;
  base_url?: string | null;
};

export const ExternalProvidersForm = memo(() => {
  const { t } = useTranslation();
  const { data, isLoading } = useGetExternalProviderConfigsQuery();
  const { data: starterModels } = useGetStarterModelsQuery();
  const [installModel] = useInstallModel();
  const { getIsInstalled, buildModelInstallArg } = useBuildModelInstallArg();

  const externalModelsByProvider = useMemo(() => {
    const groups = new Map<string, StarterModel[]>();
    for (const model of starterModels?.starter_models ?? []) {
      if (!model.source.startsWith('external://')) {
        continue;
      }
      const providerId = model.source.slice('external://'.length).split('/')[0];
      if (!providerId) {
        continue;
      }
      const models = groups.get(providerId) ?? [];
      models.push(model);
      groups.set(providerId, models);
    }

    for (const [providerId, models] of groups.entries()) {
      models.sort((a, b) => a.name.localeCompare(b.name));
      groups.set(providerId, models);
    }

    return groups;
  }, [starterModels]);

  const handleInstallProviderModels = useCallback(
    (providerId: string) => {
      const models = externalModelsByProvider.get(providerId);
      if (!models?.length) {
        return;
      }
      const modelsToInstall = models.filter((model) => !getIsInstalled(model)).map(buildModelInstallArg);
      modelsToInstall.forEach((model) => installModel(model));
    },
    [buildModelInstallArg, externalModelsByProvider, getIsInstalled, installModel]
  );

  const sortedProviders = useMemo(() => {
    if (!data) {
      return [];
    }
    return [...data].sort((a, b) => {
      const aIndex = PROVIDER_SORT_ORDER.indexOf(a.provider_id);
      const bIndex = PROVIDER_SORT_ORDER.indexOf(b.provider_id);
      if (aIndex === -1 && bIndex === -1) {
        return a.provider_id.localeCompare(b.provider_id);
      }
      if (aIndex === -1) {
        return 1;
      }
      if (bIndex === -1) {
        return -1;
      }
      return aIndex - bIndex;
    });
  }, [data]);

  return (
    <Flex flexDir="column" height="100%" gap={4}>
      <Flex flexDir="column" gap={1}>
        <Heading size="md">{t('modelManager.externalSetupTitle')}</Heading>
        <Text variant="subtext">{t('modelManager.externalSetupDescription')}</Text>
        <Text variant="subtext">{t('modelManager.externalSetupFooter')}</Text>
      </Flex>
      <ScrollableContent>
        <Flex flexDir="column" gap={4}>
          {isLoading && <Text color="base.300">{t('common.loading')}</Text>}
          {!isLoading && sortedProviders.length === 0 && (
            <Text color="base.300">{t('modelManager.externalProvidersUnavailable')}</Text>
          )}
          {sortedProviders.map((provider) => (
            <ProviderCard
              key={provider.provider_id}
              provider={provider}
              iconResolver={resolveProviderIcon}
              onInstallModels={handleInstallProviderModels}
            />
          ))}
        </Flex>
      </ScrollableContent>
    </Flex>
  );
});

ExternalProvidersForm.displayName = 'ExternalProvidersForm';

const ProviderCard = memo(({ provider, onInstallModels, iconResolver }: ProviderCardProps) => {
  const { t } = useTranslation();
  const toast = useToast();
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState(provider.base_url ?? '');
  const [saveConfig, { isLoading }] = useSetExternalProviderConfigMutation();
  const [resetConfig, { isLoading: isResetting }] = useResetExternalProviderConfigMutation();
  const [overrideBaseUrl, setOverrideBaseUrl] = useState(!!provider.base_url);

  useEffect(() => {
    setApiKey('');
    setBaseUrl(provider.base_url ?? '');
    setOverrideBaseUrl(!!provider.base_url);
  }, [provider.base_url, provider.provider_id]);

  const hasBaseUrlChange = useMemo(() => {
    if (!overrideBaseUrl) {
      return provider.base_url !== null;
    }
    return baseUrl.trim() !== (provider.base_url ?? '');
  }, [baseUrl, overrideBaseUrl, provider.base_url]);

  const handleSave = useCallback(() => {
    const trimmedApiKey = apiKey.trim();
    const trimmedBaseUrl = baseUrl.trim();
    const updatePayload: UpdatePayload = {
      provider_id: provider.provider_id,
    };
    if (trimmedApiKey) {
      updatePayload.api_key = trimmedApiKey;
    }
    if (!overrideBaseUrl && provider.base_url !== null) {
      updatePayload.base_url = null;
    } else if (overrideBaseUrl && trimmedBaseUrl !== (provider.base_url ?? '')) {
      updatePayload.base_url = trimmedBaseUrl;
    }

    if (!updatePayload.api_key && updatePayload.base_url === undefined) {
      return;
    }

    saveConfig(updatePayload)
      .unwrap()
      .then((result) => {
        if (trimmedApiKey && result.api_key_configured) {
          setApiKey('');
          onInstallModels(provider.provider_id);
        }
        setBaseUrl(result.base_url ?? '');
        setOverrideBaseUrl(!!result.base_url);
      })
      .catch(() => {
        toast({
          id: `EXTERNAL_PROVIDER_SAVE_FAILED_${provider.provider_id}`,
          title: t('modelManager.externalProviderSaveFailed'),
          status: 'error',
        });
      });
  }, [
    apiKey,
    baseUrl,
    onInstallModels,
    overrideBaseUrl,
    provider.base_url,
    provider.provider_id,
    saveConfig,
    t,
    toast,
  ]);

  const handleReset = useCallback(() => {
    resetConfig(provider.provider_id)
      .unwrap()
      .then((result) => {
        setApiKey('');
        setBaseUrl(result.base_url ?? '');
        setOverrideBaseUrl(!!result.base_url);
      })
      .catch(() => {
        toast({
          id: `EXTERNAL_PROVIDER_RESET_FAILED_${provider.provider_id}`,
          title: t('modelManager.externalProviderResetFailed'),
          status: 'error',
        });
      });
  }, [provider.provider_id, resetConfig, t, toast]);

  const handleApiKeyChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setApiKey(event.target.value);
  }, []);

  const handleBaseUrlChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setBaseUrl(event.target.value);
  }, []);

  const statusBadge = provider.api_key_configured ? (
    <Badge colorScheme="green" display="flex" alignItems="center" gap={2}>
      <PiCheckBold />
      {t('settings.externalProviderConfigured')}
    </Badge>
  ) : (
    <Badge colorScheme="warning" display="flex" alignItems="center" gap={2}>
      <PiWarningBold />
      {t('settings.externalProviderNotConfigured')}
    </Badge>
  );

  const handleOverrideBaseUrlChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    event.stopPropagation();
    setOverrideBaseUrl(event.target.checked);
    if (!event.target.checked) {
      setBaseUrl('');
    }
  }, []);

  const ProviderIcon = iconResolver(provider.provider_id);

  return (
    <Card p={4} gap={2} layerStyle="second">
      <Flex justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={3}>
        <Flex alignItems="start" gap="4">
          {ProviderIcon && <ProviderIcon />}
          <Flex flexDir="column" gap={1} mt="-0.5">
            <Heading size="xs" textTransform="capitalize" display="flex" alignItems="center" gap={2}>
              {provider.provider_id}
            </Heading>
            <Text variant="subtext">
              {t('modelManager.externalProviderCardDescription', { providerId: provider.provider_id })}
            </Text>
          </Flex>
        </Flex>
        {statusBadge}
      </Flex>
      <Flex flexDir="column" gap={4}>
        <FormControl sx={FORM_CONTROL_SX}>
          <FormLabel>{t('modelManager.externalApiKey')}</FormLabel>
          <Input
            type="password"
            autoComplete="off"
            placeholder={
              provider.api_key_configured
                ? t('modelManager.externalApiKeyPlaceholderSet')
                : t('modelManager.externalApiKeyPlaceholder')
            }
            value={apiKey}
            onChange={handleApiKeyChange}
          />
          <FormHelperText>{t('modelManager.externalApiKeyHelper')}</FormHelperText>
        </FormControl>
        <FormControl display="flex" alignItems="center">
          <Switch
            id={`${provider.provider_id}-override-baseurl`}
            isChecked={overrideBaseUrl}
            onChange={handleOverrideBaseUrlChange}
          />
          <FormLabel htmlFor={`${provider.provider_id}-override-baseurl`}>
            {t('modelManager.externalOverrideBaseUrl')}
          </FormLabel>
        </FormControl>
        <Flex hidden={!overrideBaseUrl}>
          <FormControl sx={FORM_CONTROL_SX}>
            <FormLabel>{t('modelManager.externalBaseUrl')}</FormLabel>
            <Input
              placeholder={t('modelManager.externalBaseUrlPlaceholder')}
              value={baseUrl}
              onChange={handleBaseUrlChange}
            />
            <FormHelperText>{t('modelManager.externalBaseUrlHelper')}</FormHelperText>
          </FormControl>
        </Flex>
        <Flex gap={2} justifyContent="flex-end" flexWrap="wrap" borderTopWidth="1px" pt="4">
          <Tooltip label={t('modelManager.externalResetHelper')}>
            <Button variant="ghost" onClick={handleReset} isLoading={isResetting}>
              {t('common.reset')}
            </Button>
          </Tooltip>
          <Button
            colorScheme="invokeYellow"
            onClick={handleSave}
            isLoading={isLoading}
            isDisabled={!apiKey.trim() && !hasBaseUrlChange}
          >
            {t('common.save')}
          </Button>
        </Flex>
      </Flex>
      {provider.provider_id === 'fal' && <FalModelCatalog isConfigured={provider.api_key_configured} />}
    </Card>
  );
});

ProviderCard.displayName = 'ProviderCard';

type FalModelCatalogProps = {
  isConfigured: boolean;
};

const FalModelCatalog = memo(({ isConfigured }: FalModelCatalogProps) => {
  const { t } = useTranslation();
  const toast = useToast();
  const [search, setSearch] = useState('');
  const [fetchModels, { isFetching, isError }] = useLazyGetFalModelsQuery();
  const [models, setModels] = useState<FalCatalogModel[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [installModel, { isLoading: isInstalling }] = useInstallFalModelMutation();

  const loadModels = useCallback(
    (append: boolean, cursor?: string | null) => {
      if (!isConfigured) {
        return Promise.resolve();
      }
      return fetchModels({ limit: 30, cursor: cursor ?? undefined, search: search.trim() || undefined })
        .unwrap()
        .then((result) => {
          setModels((current) => (append ? [...current, ...result.models] : result.models));
          setNextCursor(result.next_cursor);
        })
        .catch(() => undefined);
    },
    [fetchModels, isConfigured, search]
  );

  useEffect(() => {
    setModels([]);
    setNextCursor(null);
    const timer = window.setTimeout(() => void loadModels(false), 300);
    return () => window.clearTimeout(timer);
  }, [loadModels]);

  const handleInstall = useCallback(
    (model: FalCatalogModel) => {
      installModel({ endpoint_id: model.endpoint_id })
        .unwrap()
        .catch(() => {
          toast({
            id: `FAL_MODEL_INSTALL_FAILED_${model.endpoint_id}`,
            title: t('modelManager.externalProviderSaveFailed'),
            status: 'error',
          });
        });
    },
    [installModel, t, toast]
  );

  if (!isConfigured) {
    return null;
  }

  return (
    <Flex flexDir="column" gap={3} borderTopWidth="1px" pt={4}>
      <Flex alignItems="center" justifyContent="space-between" gap={2} flexWrap="wrap">
        <Flex flexDir="column" gap={1}>
          <Heading size="xs">{t('modelManager.externalFalCatalogTitle')}</Heading>
          <Text variant="subtext">{t('modelManager.externalFalCatalogDescription')}</Text>
        </Flex>
        <Button variant="ghost" onClick={() => void loadModels(false)} isLoading={isFetching}>
          {t('modelManager.externalFalCatalogRefresh')}
        </Button>
      </Flex>
      <Input
        placeholder={t('modelManager.externalFalCatalogSearch')}
        value={search}
        onChange={(event) => setSearch(event.target.value)}
      />
      {isError && <Text color="base.300">{t('modelManager.externalFalCatalogLoadFailed')}</Text>}
      {isFetching && models.length === 0 && <Text color="base.300">{t('modelManager.externalFalCatalogLoading')}</Text>}
      {models.map((model) => {
        const isCanvasModel = isFalNativeCanvasModel(model);
        return (
          <Flex key={model.endpoint_id} alignItems="center" justifyContent="space-between" gap={3} flexWrap="wrap">
            <Flex flexDir="column" gap={1} minW={0}>
              <Flex alignItems="center" gap={2} flexWrap="wrap">
                <Text fontWeight="semibold">{model.display_name}</Text>
                <Badge>{model.kind}</Badge>
              </Flex>
              <Text variant="subtext" noOfLines={1} title={model.endpoint_id}>
                {model.endpoint_id}
              </Text>
            </Flex>
            {isCanvasModel ? (
              <Button
                size="sm"
                onClick={() => handleInstall(model)}
                isDisabled={model.installed}
                isLoading={isInstalling}
              >
                {model.installed
                  ? t('modelManager.externalFalCatalogInstalled')
                  : t('modelManager.externalFalCatalogInstall')}
              </Button>
            ) : (
              <Badge colorScheme="gray">{t('modelManager.externalFalCatalogGeneric')}</Badge>
            )}
          </Flex>
        );
      })}
      {nextCursor && (
        <Button variant="ghost" onClick={() => void loadModels(true, nextCursor)} isLoading={isFetching}>
          {t('common.loadMore')}
        </Button>
      )}
    </Flex>
  );
});

FalModelCatalog.displayName = 'FalModelCatalog';
