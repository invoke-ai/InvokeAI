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
import { PiCheckBold, PiWarningBold } from 'react-icons/pi';
import { SiAlibabacloud, SiBytedance, SiGooglegemini, SiOpenai } from 'react-icons/si';
import {
  useGetExternalProviderConfigsQuery,
  useResetExternalProviderConfigMutation,
  useSetExternalProviderConfigMutation,
} from 'services/api/endpoints/appInfo';
import { useGetStarterModelsQuery } from 'services/api/endpoints/models';
import type { ExternalProviderConfig, StarterModel } from 'services/api/types';

const PROVIDER_SORT_ORDER = ['gemini', 'openai', 'seedream', 'alibabacloud'];

function resolveProviderIcon(providerId: string): IconType | null {
  const provider = providerId.toLowerCase();

  switch (provider) {
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
    </Card>
  );
});

ProviderCard.displayName = 'ProviderCard';
