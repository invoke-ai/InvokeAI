/* eslint-disable react/react-compiler, react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { HFTokenStatus } from '@features/models/core/types';

import { Badge, Flex, Grid, HStack, Icon, Input, Stack, Switch, Text } from '@chakra-ui/react';
import {
  getExternalProviderConfigs,
  getHFTokenStatus,
  resetExternalProviderConfig,
  resetHFToken,
  setExternalProviderConfig,
  setHFToken,
  type ExternalProviderConfig,
} from '@features/models/data/api';
import { clearCivitaiApiKey, getCivitaiApiKey, setCivitaiApiKey } from '@features/models/data/apiKeys';
import { refreshInstalls } from '@features/models/data/installsStore';
import { refreshStartersIfLoaded } from '@features/models/data/startersStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { Button } from '@platform/ui';
import { HexagonIcon } from 'lucide-react';
import { useEffect, useState, type ElementType } from 'react';
import { useTranslation } from 'react-i18next';
import { SiAlibabacloud, SiBytedance, SiGooglegemini, SiHuggingface, SiOpenai } from 'react-icons/si';

/**
 * Credentials for every model source, as one uniform grid of key cards:
 * HuggingFace (server-stored, verified), Civitai (browser-local), and the
 * external image providers (server config). Each card shares the same
 * edit/display anatomy: icon, status, prefix-hinted input, save/clear.
 */
export const ApiKeysSection = () => {
  const notify = useNotify();
  const { t } = useTranslation();

  return (
    <Stack gap="3">
      <Text color="fg.subtle" fontSize="2xs">
        {t('models.apiKeysDescription')}
      </Text>
      <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(19rem, 1fr))">
        <HuggingFaceKeyCard onError={notify.error} />
        <CivitaiKeyCard onError={(message) => notify.error(t('models.civitaiKey'), message)} />
        <ExternalProviderKeyCards onError={notify.error} />
      </Grid>
    </Stack>
  );
};

interface KeyStatusBadge {
  label: string;
  palette: string;
}

/** The shared card: identical anatomy whether the key is set or not. */
const ApiKeyCard = ({
  description,
  icon,
  isLoading = false,
  onClear,
  onError,
  onSave,
  placeholder,
  status,
  title,
}: {
  description: string;
  icon: ElementType;
  isLoading?: boolean;
  /** Absent until a key exists; clears the stored key. */
  onClear?: () => Promise<void> | void;
  /** Save/clear failures (network, backend rejection) land here. */
  onError: (message: string) => void;
  onSave: (key: string) => Promise<void> | void;
  /** Hint at the expected shape, e.g. "hf_…" or "sk-…". */
  placeholder: string;
  status: KeyStatusBadge | null;
  title: string;
}) => {
  const { t } = useTranslation();
  const [draft, setDraft] = useState('');
  const [isBusy, setIsBusy] = useState(false);

  const runAction = async (action: () => Promise<void> | void): Promise<boolean> => {
    setIsBusy(true);

    try {
      await action();

      return true;
    } catch (error) {
      onError(error instanceof Error ? error.message : t('common.somethingWentWrong'));

      return false;
    } finally {
      setIsBusy(false);
    }
  };

  const handleSave = async () => {
    const key = draft.trim();

    if (key.length === 0) {
      return;
    }

    // Keep the draft on failure so the key can be corrected and retried.
    if (await runAction(() => onSave(key))) {
      setDraft('');
    }
  };

  return (
    <Stack bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" gap="2.5" p="3" rounded="lg">
      <HStack align="start" gap="2.5">
        <Flex
          align="center"
          bg="bg.emphasized"
          borderColor="border.subtle"
          borderWidth="1px"
          boxSize="8"
          color="fg.muted"
          flexShrink={0}
          justify="center"
          rounded="md"
        >
          <Icon as={icon} boxSize="4" />
        </Flex>
        <Stack flex="1" gap="0" minW="0">
          <HStack gap="1.5">
            <Text fontSize="xs" fontWeight="700" truncate>
              {title}
            </Text>
            {status ? (
              <Badge colorPalette={status.palette} flexShrink={0} fontSize="2xs" size="sm" variant="surface">
                {status.label}
              </Badge>
            ) : null}
          </HStack>
          <Text color="fg.subtle" fontSize="2xs" lineClamp={2}>
            {description}
          </Text>
        </Stack>
      </HStack>
      <HStack gap="1.5">
        <Input
          aria-label={t('models.apiKeyFor', { title })}
          disabled={isLoading}
          placeholder={placeholder}
          size="xs"
          type="password"
          value={draft}
          onChange={(event) => setDraft(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              void handleSave();
            }
          }}
        />
        <Button
          disabled={draft.trim().length === 0 || isLoading}
          loading={isBusy && draft.trim().length > 0}
          size="xs"
          variant="solid"
          onClick={() => void handleSave()}
        >
          {t('common.save')}
        </Button>
        {onClear ? (
          <Button disabled={isBusy || isLoading} size="sm" variant="ghost" onClick={() => void runAction(onClear)}>
            {t('common.clear')}
          </Button>
        ) : null}
      </HStack>
    </Stack>
  );
};

const HF_STATUS_PALETTES: Record<HFTokenStatus, string> = {
  invalid: 'red',
  unknown: 'gray',
  valid: 'green',
};

const HuggingFaceKeyCard = ({ onError }: { onError: (title: string, message: string) => void }) => {
  const { t } = useTranslation();
  const [status, setStatus] = useState<HFTokenStatus | null>(null);
  const statusBadge = status ? { label: t(`models.keyStatus.${status}`), palette: HF_STATUS_PALETTES[status] } : null;

  useEffect(() => {
    let isStale = false;

    getHFTokenStatus()
      .then((tokenStatus) => {
        if (!isStale) {
          setStatus(tokenStatus);
        }
      })
      .catch(() => {
        if (!isStale) {
          setStatus('unknown');
        }
      });

    return () => {
      isStale = true;
    };
  }, [t]);

  return (
    <ApiKeyCard
      description={t('models.huggingFaceKeyDescription')}
      icon={SiHuggingface}
      isLoading={status === null}
      placeholder="hf_…"
      status={statusBadge}
      title="HuggingFace"
      onError={(message) => onError(t('models.huggingFaceToken'), message)}
      onClear={
        status === 'valid' || status === 'invalid'
          ? async () => {
              setStatus(await resetHFToken());
            }
          : undefined
      }
      onSave={async (key) => {
        const nextStatus = await setHFToken(key);

        setStatus(nextStatus);

        if (nextStatus === 'invalid') {
          onError(t('models.huggingFaceToken'), t('models.huggingFaceRejectedToken'));
        }
      }}
    />
  );
};

const CivitaiKeyCard = ({ onError }: { onError: (message: string) => void }) => {
  const { t } = useTranslation();
  const [hasKey, setHasKey] = useState(() => getCivitaiApiKey() !== null);

  return (
    <ApiKeyCard
      description={t('models.civitaiKeyDescription')}
      icon={HexagonIcon}
      placeholder={t('models.civitaiKeyPlaceholder')}
      status={
        hasKey
          ? { label: t('models.keyStatus.saved'), palette: 'green' }
          : { label: t('models.keyStatus.unknown'), palette: 'gray' }
      }
      title="Civitai"
      onError={onError}
      onClear={
        hasKey
          ? () => {
              clearCivitaiApiKey();
              setHasKey(false);
            }
          : undefined
      }
      onSave={(key) => {
        setCivitaiApiKey(key);
        setHasKey(true);
      }}
    />
  );
};

const EXTERNAL_PROVIDER_PRESENTATION: Record<string, { title: string; icon: ElementType; placeholder: string }> = {
  alibabacloud: { icon: SiAlibabacloud, placeholder: 'sk-…', title: 'Alibaba Cloud (Qwen)' },
  gemini: { icon: SiGooglegemini, placeholder: 'AIza…', title: 'Google Gemini' },
  openai: { icon: SiOpenai, placeholder: 'sk-…', title: 'OpenAI' },
  seedream: { icon: SiBytedance, placeholder: 'byteplus', title: 'Seedream' },
};

const ExternalProviderKeyCards = ({ onError }: { onError: (title: string, message: string) => void }) => {
  const { t } = useTranslation();
  const [configs, setConfigs] = useState<ExternalProviderConfig[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let isStale = false;

    getExternalProviderConfigs()
      .then((providerConfigs) => {
        if (!isStale) {
          setConfigs(providerConfigs);
        }
      })
      .catch((error: unknown) => {
        if (!isStale) {
          setConfigs([]);
          setLoadError(error instanceof Error ? error.message : t('models.failedToLoadExternalProviders'));
        }
      });

    return () => {
      isStale = true;
    };
  }, [t]);

  const replaceConfig = (next: ExternalProviderConfig) => {
    setConfigs((current) => (current ?? []).map((config) => (config.provider_id === next.provider_id ? next : config)));
  };

  if (loadError) {
    return (
      <Text color="fg.error" fontSize="2xs">
        {t('models.externalProvidersUnavailable', { error: loadError })}
      </Text>
    );
  }

  return (
    <>
      {(configs ?? []).map((config) => {
        const presentation = EXTERNAL_PROVIDER_PRESENTATION[config.provider_id] ?? {
          icon: HexagonIcon,
          placeholder: 'generic',
          title: config.provider_id,
        };
        const placeholder =
          presentation.placeholder === 'byteplus'
            ? t('models.bytePlusApiKeyPlaceholder')
            : presentation.placeholder === 'generic'
              ? t('models.apiKey')
              : presentation.placeholder;

        return (
          <ExternalProviderKeyCard
            config={config}
            key={config.provider_id}
            description={t('models.externalProviderKeyDescription')}
            icon={presentation.icon}
            placeholder={placeholder}
            title={presentation.title}
            onError={(message) => onError(presentation.title, message)}
            onUpdated={replaceConfig}
          />
        );
      })}
    </>
  );
};

const ExternalProviderKeyCard = ({
  config,
  description,
  icon,
  onError,
  onUpdated,
  placeholder,
  title,
}: {
  config: ExternalProviderConfig;
  description: string;
  icon: ElementType;
  onError: (message: string) => void;
  onUpdated: (next: ExternalProviderConfig) => void;
  placeholder: string;
  title: string;
}) => {
  const { t } = useTranslation();
  const [apiKeyDraft, setApiKeyDraft] = useState('');
  const [baseUrlDraft, setBaseUrlDraft] = useState(config.base_url ?? '');
  const [overrideBaseUrl, setOverrideBaseUrl] = useState(config.base_url !== null);
  const [isBusy, setIsBusy] = useState(false);

  useEffect(() => {
    setApiKeyDraft('');
    setBaseUrlDraft(config.base_url ?? '');
    setOverrideBaseUrl(config.base_url !== null);
  }, [config.base_url, config.provider_id]);

  const hasApiKeyDraft = apiKeyDraft.trim().length > 0;
  const hasBaseUrlChange = overrideBaseUrl ? baseUrlDraft.trim() !== (config.base_url ?? '') : config.base_url !== null;

  const runAction = async (
    action: () => Promise<ExternalProviderConfig>,
    onSuccess?: (next: ExternalProviderConfig) => void
  ): Promise<void> => {
    setIsBusy(true);

    try {
      const nextConfig = await action();

      onUpdated(nextConfig);
      setApiKeyDraft('');
      setBaseUrlDraft(nextConfig.base_url ?? '');
      setOverrideBaseUrl(nextConfig.base_url !== null);
      onSuccess?.(nextConfig);
    } catch (error) {
      onError(error instanceof Error ? error.message : t('common.somethingWentWrong'));
    } finally {
      setIsBusy(false);
    }
  };

  const handleSave = async () => {
    const apiKey = apiKeyDraft.trim();
    const baseUrl = baseUrlDraft.trim();
    const nextConfig: { api_key?: string; base_url?: string | null } = {};

    if (apiKey.length > 0) {
      nextConfig.api_key = apiKey;
    }

    if (!overrideBaseUrl && config.base_url !== null) {
      nextConfig.base_url = '';
    } else if (overrideBaseUrl && baseUrl !== (config.base_url ?? '')) {
      nextConfig.base_url = baseUrl;
    }

    if (!nextConfig.api_key && nextConfig.base_url === undefined) {
      return;
    }

    const apiKeyWasSet = nextConfig.api_key !== undefined;

    await runAction(
      () => setExternalProviderConfig(config.provider_id, nextConfig),
      (next) => {
        if (apiKeyWasSet && next.api_key_configured) {
          // The backend queues this provider's external starter models on
          // key-set; pull the new jobs and refresh starters so its rows flip to
          // Installed without an app restart.
          void refreshInstalls();
          refreshStartersIfLoaded();
        }
      }
    );
  };

  const canClear = config.api_key_configured || config.base_url !== null;

  return (
    <Stack bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" gap="2.5" p="3" rounded="lg">
      <HStack align="start" gap="2.5">
        <Flex
          align="center"
          bg="bg.emphasized"
          borderColor="border.subtle"
          borderWidth="1px"
          boxSize="8"
          color="fg.muted"
          flexShrink={0}
          justify="center"
          rounded="md"
        >
          <Icon as={icon} boxSize="4" />
        </Flex>
        <Stack flex="1" gap="0" minW="0">
          <HStack gap="1.5">
            <Text fontSize="xs" fontWeight="700" truncate>
              {title}
            </Text>
            <Badge
              colorPalette={config.api_key_configured ? 'green' : 'gray'}
              flexShrink={0}
              fontSize="2xs"
              size="sm"
              variant="surface"
            >
              {config.api_key_configured ? t('models.keyStatus.configured') : t('models.keyStatus.unknown')}
            </Badge>
          </HStack>
          <Text color="fg.subtle" fontSize="2xs" lineClamp={2}>
            {description}
          </Text>
        </Stack>
      </HStack>

      <Stack gap="2">
        <HStack gap="1.5">
          <Input
            aria-label={t('models.apiKeyFor', { title })}
            disabled={isBusy}
            placeholder={config.api_key_configured ? t('models.apiKeyConfigured') : placeholder}
            size="xs"
            type="password"
            value={apiKeyDraft}
            onChange={(event) => setApiKeyDraft(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                void handleSave();
              }
            }}
          />
        </HStack>
        <Switch.Root
          checked={overrideBaseUrl}
          disabled={isBusy}
          size="sm"
          onCheckedChange={(event) => {
            const checked = event.checked === true;

            setOverrideBaseUrl(checked);

            if (!checked) {
              setBaseUrlDraft('');
            }
          }}
        >
          <Switch.HiddenInput />
          <Switch.Control _checked={{ bg: 'accent.solid' }}>
            <Switch.Thumb />
          </Switch.Control>
          <Switch.Label color="fg.muted" fontSize="2xs">
            {t('models.overrideBaseUrl')}
          </Switch.Label>
        </Switch.Root>
        {overrideBaseUrl ? (
          <Input
            aria-label={t('models.baseUrlFor', { title })}
            disabled={isBusy}
            placeholder="https://api.example.com"
            size="xs"
            value={baseUrlDraft}
            onChange={(event) => setBaseUrlDraft(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                void handleSave();
              }
            }}
          />
        ) : null}
      </Stack>

      <HStack gap="1.5">
        <Button
          disabled={(!hasApiKeyDraft && !hasBaseUrlChange) || isBusy}
          loading={isBusy}
          size="xs"
          variant="solid"
          onClick={() => void handleSave()}
        >
          {t('common.save')}
        </Button>
        {canClear ? (
          <Button
            disabled={isBusy}
            size="sm"
            variant="ghost"
            onClick={() => void runAction(() => resetExternalProviderConfig(config.provider_id))}
          >
            {t('common.clear')}
          </Button>
        ) : null}
      </HStack>
    </Stack>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
