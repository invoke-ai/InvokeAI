import {
  Badge,
  Button,
  Card,
  ConfirmationAlertDialog,
  Flex,
  FormControl,
  FormHelperText,
  FormLabel,
  Heading,
  IconButton,
  Input,
  Text,
  Tooltip,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useBuildModelInstallArg } from 'features/modelManagerV2/hooks/useBuildModelsToInstall';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import { $installModelsTabIndex } from 'features/modelManagerV2/store/installModelsStore';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold, PiPlusBold, PiTrashBold, PiWarningBold } from 'react-icons/pi';
import {
  useCreateCustomOpenAIImagesModelMutation,
  useDeleteCustomOpenAIImagesModelMutation,
  useGetCustomOpenAIImagesModelsQuery,
  useGetExternalProviderConfigsQuery,
  useResetExternalProviderConfigMutation,
  useSetExternalProviderConfigMutation,
} from 'services/api/endpoints/appInfo';
import { useGetStarterModelsQuery } from 'services/api/endpoints/models';
import type { ExternalApiModelConfig, ExternalProviderConfig, StarterModel } from 'services/api/types';

const CUSTOM_OPENAI_IMAGES_PROVIDER_ID = 'custom_openai_images';
const PROVIDER_SORT_ORDER = ['gemini', 'openai', CUSTOM_OPENAI_IMAGES_PROVIDER_ID, 'seedream', 'alibabacloud'];

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
  const tabIndex = useStore($installModelsTabIndex);

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
        <Heading size="sm">{t('modelManager.externalSetupTitle')}</Heading>
        <Text color="base.300">{t('modelManager.externalSetupDescription')}</Text>
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
  const [pendingSavePayload, setPendingSavePayload] = useState<UpdatePayload | null>(null);
  const [saveConfig, { isLoading }] = useSetExternalProviderConfigMutation();
  const [resetConfig, { isLoading: isResetting }] = useResetExternalProviderConfigMutation();
  const isCustomOpenAIImages = provider.provider_id === CUSTOM_OPENAI_IMAGES_PROVIDER_ID;
  const { data: customModels } = useGetCustomOpenAIImagesModelsQuery(undefined, { skip: !isCustomOpenAIImages });
  const {
    isOpen: isBaseUrlWarningOpen,
    onOpen: onBaseUrlWarningOpen,
    onClose: onBaseUrlWarningClose,
  } = useDisclosure();

  useEffect(() => {
    setBaseUrl(provider.base_url ?? '');
  }, [provider.base_url]);

  const providerLabel = useMemo(
    () =>
      isCustomOpenAIImages
        ? t('modelManager.customOpenAIImagesProviderTitle', 'Custom OpenAI Images-compatible')
        : `${provider.provider_id.charAt(0).toUpperCase()}${provider.provider_id.slice(1)}`,
    [isCustomOpenAIImages, provider.provider_id, t]
  );

  const executeSave = useCallback(
    (updatePayload: UpdatePayload) => {
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
          setPendingSavePayload(null);
        });
    },
    [onInstallModels, provider.provider_id, saveConfig]
  );

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

    if (isCustomOpenAIImages && updatePayload.base_url !== undefined && (customModels?.length ?? 0) > 0) {
      setPendingSavePayload(updatePayload);
      onBaseUrlWarningOpen();
      return;
    }

    executeSave(updatePayload);
  }, [
    apiKey,
    baseUrl,
    customModels?.length,
    executeSave,
    isCustomOpenAIImages,
    onBaseUrlWarningOpen,
    provider.base_url,
    provider.provider_id,
  ]);

  const handleConfirmBaseUrlChange = useCallback(() => {
    if (pendingSavePayload) {
      executeSave(pendingSavePayload);
    }
    onBaseUrlWarningClose();
  }, [executeSave, onBaseUrlWarningClose, pendingSavePayload]);

  const handleCloseBaseUrlWarning = useCallback(() => {
    setPendingSavePayload(null);
    onBaseUrlWarningClose();
  }, [onBaseUrlWarningClose]);

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

  const isProviderConfigured = provider.api_key_configured && (!isCustomOpenAIImages || Boolean(provider.base_url));
  const statusBadge = isProviderConfigured ? (
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
          <Heading size="xs">{providerLabel}</Heading>
          <Text variant="subtext">
            {isCustomOpenAIImages
              ? t(
                  'modelManager.customOpenAIImagesProviderDescription',
                  'Configure a custom OpenAI Images-compatible endpoint and add its models manually.'
                )
              : t('modelManager.externalProviderCardDescription', { providerId: provider.provider_id })}
          </Text>
        </Flex>
        {statusBadge}
      </Flex>
      <Flex flexDir="column" gap={4}>
        <FormControl>
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
        <FormControl>
          <FormLabel>
            {isCustomOpenAIImages
              ? t('modelManager.customOpenAIImagesBaseUrl', 'Base URL')
              : t('modelManager.externalBaseUrl')}
          </FormLabel>
          <Input
            placeholder={t('modelManager.externalBaseUrlPlaceholder')}
            value={baseUrl}
            onChange={handleBaseUrlChange}
          />
          <FormHelperText>
            {isCustomOpenAIImages
              ? t(
                  'modelManager.customOpenAIImagesBaseUrlHelper',
                  'Required. Use either the provider root URL or a /v1 OpenAI-compatible base URL.'
                )
              : t('modelManager.externalBaseUrlHelper')}
          </FormHelperText>
        </FormControl>
        {isCustomOpenAIImages && (
          <CustomOpenAIImagesModelsSection
            models={customModels ?? []}
            isConnectionReady={provider.api_key_configured && Boolean(baseUrl.trim())}
          />
        )}
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
      {isCustomOpenAIImages && (
        <ConfirmationAlertDialog
          isOpen={isBaseUrlWarningOpen}
          onClose={handleCloseBaseUrlWarning}
          title={t('modelManager.customOpenAIImagesBaseUrlWarningTitle', 'Change custom provider URL?')}
          acceptCallback={handleConfirmBaseUrlChange}
          acceptButtonText={t('common.confirm', 'Confirm')}
          useInert={false}
        >
          <Text>
            {t(
              'modelManager.customOpenAIImagesBaseUrlWarning',
              'Saved custom models may stop working if this connection starts pointing at another provider.'
            )}
          </Text>
        </ConfirmationAlertDialog>
      )}
    </Card>
  );
});

ProviderCard.displayName = 'ProviderCard';

const CustomOpenAIImagesModelsSection = memo(
  ({ models, isConnectionReady }: { models: ExternalApiModelConfig[]; isConnectionReady: boolean }) => {
    const { t } = useTranslation();
    const [modelName, setModelName] = useState('');
    const [providerModelId, setProviderModelId] = useState('');
    const [createModel, { isLoading: isCreating }] = useCreateCustomOpenAIImagesModelMutation();
    const [deleteModel, { isLoading: isDeleting }] = useDeleteCustomOpenAIImagesModelMutation();

    const handleModelNameChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
      setModelName(event.target.value);
    }, []);

    const handleProviderModelIdChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
      setProviderModelId(event.target.value);
    }, []);

    const handleAddModel = useCallback(() => {
      const trimmedProviderModelId = providerModelId.trim();
      if (!trimmedProviderModelId) {
        return;
      }
      createModel({
        provider_model_id: trimmedProviderModelId,
        name: modelName.trim() || null,
      })
        .unwrap()
        .then(() => {
          setModelName('');
          setProviderModelId('');
        });
    }, [createModel, modelName, providerModelId]);

    return (
      <Flex borderTopWidth={1} borderColor="base.700" pt={4} flexDir="column" gap={3}>
        <Flex justifyContent="space-between" alignItems="center" gap={3} flexWrap="wrap">
          <Flex flexDir="column" gap={1}>
            <Heading size="xs">{t('modelManager.customOpenAIImagesModelsTitle', 'Custom models')}</Heading>
            <Text variant="subtext">
              {t(
                'modelManager.customOpenAIImagesModelsDescription',
                'Add provider model IDs exactly as they are documented by the API provider.'
              )}
            </Text>
          </Flex>
        </Flex>
        <Flex gap={3} alignItems="flex-end" flexWrap="wrap">
          <FormControl flex="1 1 14rem">
            <FormLabel>{t('modelManager.customOpenAIImagesModelId', 'Provider Model ID')}</FormLabel>
            <Input
              placeholder={t('modelManager.customOpenAIImagesModelIdPlaceholder', 'google/gemini-2.5-flash-image')}
              value={providerModelId}
              onChange={handleProviderModelIdChange}
            />
          </FormControl>
          <FormControl flex="1 1 14rem">
            <FormLabel>{t('modelManager.customOpenAIImagesModelName', 'Display Name')}</FormLabel>
            <Input
              placeholder={t('modelManager.customOpenAIImagesModelNamePlaceholder', 'Gemini image via provider')}
              value={modelName}
              onChange={handleModelNameChange}
            />
          </FormControl>
          <Button
            leftIcon={<PiPlusBold />}
            onClick={handleAddModel}
            isLoading={isCreating}
            isDisabled={!isConnectionReady || !providerModelId.trim()}
            flexShrink={0}
          >
            {t('common.add', 'Add')}
          </Button>
        </Flex>
        {models.length > 0 ? (
          <Flex flexDir="column" gap={2}>
            {models.map((model) => (
              <CustomOpenAIImagesModelListItem
                key={model.key}
                model={model}
                onDelete={deleteModel}
                isDeleting={isDeleting}
              />
            ))}
          </Flex>
        ) : (
          <Text variant="subtext" color="base.400">
            {t('modelManager.customOpenAIImagesNoModels', 'No custom models added yet.')}
          </Text>
        )}
      </Flex>
    );
  }
);

CustomOpenAIImagesModelsSection.displayName = 'CustomOpenAIImagesModelsSection';

const CustomOpenAIImagesModelListItem = memo(
  ({
    model,
    onDelete,
    isDeleting,
  }: {
    model: ExternalApiModelConfig;
    onDelete: (modelKey: string) => void;
    isDeleting: boolean;
  }) => {
    const { t } = useTranslation();

    const handleDelete = useCallback(() => {
      onDelete(model.key);
    }, [model.key, onDelete]);

    return (
      <Flex
        alignItems="center"
        justifyContent="space-between"
        gap={3}
        borderWidth={1}
        borderColor="base.700"
        borderRadius="base"
        px={3}
        py={2}
      >
        <Flex flexDir="column" minW={0}>
          <Text fontWeight="semibold" noOfLines={1}>
            {model.name}
          </Text>
          <Text variant="subtext" noOfLines={1}>
            {model.provider_model_id}
          </Text>
        </Flex>
        <Tooltip label={t('common.delete', 'Delete')}>
          <IconButton
            aria-label={t('common.delete', 'Delete')}
            icon={<PiTrashBold />}
            size="sm"
            variant="ghost"
            colorScheme="error"
            onClick={handleDelete}
            isLoading={isDeleting}
          />
        </Tooltip>
      </Flex>
    );
  }
);

CustomOpenAIImagesModelListItem.displayName = 'CustomOpenAIImagesModelListItem';
