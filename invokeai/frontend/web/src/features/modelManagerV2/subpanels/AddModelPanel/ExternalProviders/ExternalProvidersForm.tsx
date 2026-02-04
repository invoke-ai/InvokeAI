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
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useBuildModelInstallArg } from 'features/modelManagerV2/hooks/useBuildModelsToInstall';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import { $installModelsTabIndex } from 'features/modelManagerV2/store/installModelsStore';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold, PiWarningBold } from 'react-icons/pi';
import {
  useGetExternalProviderConfigsQuery,
  useResetExternalProviderConfigMutation,
  useSetExternalProviderConfigMutation,
} from 'services/api/endpoints/appInfo';
import { useGetStarterModelsQuery } from 'services/api/endpoints/models';
import type { ExternalProviderConfig, StarterModel } from 'services/api/types';

const PROVIDER_SORT_ORDER = ['gemini', 'openai'];

type ProviderCardProps = {
  provider: ExternalProviderConfig;
  onInstallModels: (providerId: string) => void;
};

type UpdatePayload = {
  provider_id: string;
  api_key?: string;
  base_url?: string;
};

export const ExternalProvidersForm = memo(() => {
  const { t } = useTranslation();
  const { data, isLoading } = useGetExternalProviderConfigsQuery();
  const { data: starterModels } = useGetStarterModelsQuery();
  const [installModel] = useInstallModel();
  const { getIsInstalled, buildModelInstallArg } = useBuildModelInstallArg();
  const [installDefaults, setInstallDefaults] = useState(true);
  const tabIndex = useStore($installModelsTabIndex);

  const toggleInstallDefaults = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setInstallDefaults(event.target.checked);
  }, []);

  const externalModelsByProvider = useMemo(() => {
    const groups = new Map<string, StarterModel[]>();
    for (const model of starterModels?.starter_models ?? []) {
      if (!model.source.startsWith('external://')) {
        continue;
      }
      const providerId = model.source.replace('external://', '').split('/')[0];
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
      if (!installDefaults) {
        return;
      }
      const models = externalModelsByProvider.get(providerId);
      if (!models?.length) {
        return;
      }
      const modelsToInstall = models.filter((model) => !getIsInstalled(model)).map(buildModelInstallArg);
      modelsToInstall.forEach((model) => installModel(model));
    },
    [buildModelInstallArg, externalModelsByProvider, getIsInstalled, installDefaults, installModel]
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
      <Flex justifyContent="space-between" alignItems="center" gap={4} flexWrap="wrap">
        <Flex flexDir="column" gap={1}>
          <Heading size="sm">{t('modelManager.externalSetupTitle')}</Heading>
          <Text color="base.300">{t('modelManager.externalSetupDescription')}</Text>
        </Flex>
        <FormControl display="flex" alignItems="center" gap={3} w="fit-content">
          <FormLabel m={0}>{t('modelManager.externalInstallDefaults')}</FormLabel>
          <Switch isChecked={installDefaults} onChange={toggleInstallDefaults} />
        </FormControl>
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
              onInstallModels={handleInstallProviderModels}
            />
          ))}
        </Flex>
      </ScrollableContent>
      {tabIndex === 3 && (
        <Text variant="subtext" color="base.400">
          {t('modelManager.externalSetupFooter')}
        </Text>
      )}
    </Flex>
  );
});

ExternalProvidersForm.displayName = 'ExternalProvidersForm';

const ProviderCard = memo(({ provider, onInstallModels }: ProviderCardProps) => {
  const { t } = useTranslation();
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState(provider.base_url ?? '');
  const [saveConfig, { isLoading }] = useSetExternalProviderConfigMutation();
  const [resetConfig, { isLoading: isResetting }] = useResetExternalProviderConfigMutation();

  useEffect(() => {
    setBaseUrl(provider.base_url ?? '');
  }, [provider.base_url]);

  const handleSave = useCallback(() => {
    const trimmedApiKey = apiKey.trim();
    const trimmedBaseUrl = baseUrl.trim();
    const updatePayload: UpdatePayload = {
      provider_id: provider.provider_id,
    };
    if (trimmedApiKey) {
      updatePayload.api_key = trimmedApiKey;
    }
    if (trimmedBaseUrl !== (provider.base_url ?? '')) {
      updatePayload.base_url = trimmedBaseUrl;
    }

    if (!updatePayload.api_key && updatePayload.base_url === undefined) {
      return;
    }

    saveConfig(updatePayload)
      .unwrap()
      .then((result) => {
        if (result.api_key_configured) {
          setApiKey('');
          onInstallModels(provider.provider_id);
        }
        if (result.base_url !== undefined) {
          setBaseUrl(result.base_url ?? '');
        }
      });
  }, [apiKey, baseUrl, onInstallModels, provider.base_url, provider.provider_id, saveConfig]);

  const handleReset = useCallback(() => {
    resetConfig(provider.provider_id)
      .unwrap()
      .then((result) => {
        setApiKey('');
        setBaseUrl(result.base_url ?? '');
      });
  }, [provider.provider_id, resetConfig]);

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

  return (
    <Card p={4} gap={4} variant="outline">
      <Flex justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={3}>
        <Flex flexDir="column" gap={1}>
          <Heading size="xs" textTransform="capitalize">
            {provider.provider_id}
          </Heading>
          <Text variant="subtext">
            {t('modelManager.externalProviderCardDescription', { providerId: provider.provider_id })}
          </Text>
        </Flex>
        {statusBadge}
      </Flex>
      <Flex flexDir="column" gap={4}>
        <FormControl>
          <FormLabel>{t('modelManager.externalApiKey')}</FormLabel>
          <Input
            type="password"
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
        <FormControl>
          <FormLabel>{t('modelManager.externalBaseUrl')}</FormLabel>
          <Input
            placeholder={t('modelManager.externalBaseUrlPlaceholder')}
            value={baseUrl}
            onChange={handleBaseUrlChange}
          />
          <FormHelperText>{t('modelManager.externalBaseUrlHelper')}</FormHelperText>
        </FormControl>
        <Flex gap={2} justifyContent="flex-end" flexWrap="wrap">
          <Tooltip label={t('modelManager.externalResetHelper')}>
            <Button variant="ghost" onClick={handleReset} isLoading={isResetting}>
              {t('common.reset')}
            </Button>
          </Tooltip>
          <Button
            colorScheme="invokeYellow"
            onClick={handleSave}
            isLoading={isLoading}
            isDisabled={!apiKey.trim() && baseUrl.trim() === (provider.base_url ?? '')}
          >
            {t('common.save')}
          </Button>
        </Flex>
      </Flex>
    </Card>
  );
});

ProviderCard.displayName = 'ProviderCard';
