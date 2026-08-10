import type { ExternalProviderConfig } from '@features/models/data/api';
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ElementType } from 'react';

import { HStack, Input, Stack, Switch, Text } from '@chakra-ui/react';
import {
  clearExternalProviderConfig,
  ensureExternalProvidersLoaded,
  saveExternalProviderConfig,
  useExternalProvidersSelector,
} from '@features/models/data/externalProvidersStore';
import { refreshInstalls } from '@features/models/data/installsStore';
import { refreshStartersIfLoaded } from '@features/models/data/startersStore';
import { KeyCardShell } from '@features/models/ui/credentials/KeyCardShell';
import { useScopedAction } from '@features/models/ui/shared/useScopedAction';
import { useMountEffect } from '@platform/react/useMountEffect';
import { assertAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui';
import { BotIcon, HexagonIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SiAlibabacloud, SiBytedance, SiGooglegemini } from 'react-icons/si';

interface ProviderPresentation {
  icon: ElementType;
  /** Literal key-shape hint; `placeholderKey` (an i18n key) wins when set. */
  placeholder?: string;
  placeholderKey?: string;
  title: string;
}

const EXTERNAL_PROVIDER_PRESENTATION: Record<string, ProviderPresentation> = {
  alibabacloud: { icon: SiAlibabacloud, placeholder: 'sk-…', title: 'Alibaba Cloud (Qwen)' },
  gemini: { icon: SiGooglegemini, placeholder: 'AIza…', title: 'Google Gemini' },
  openai: { icon: BotIcon, placeholder: 'sk-…', title: 'OpenAI' },
  seedream: { icon: SiBytedance, placeholderKey: 'models.bytePlusApiKeyPlaceholder', title: 'Seedream' },
};

/** One key card per provider the backend reports, from the shared store. */
export const ExternalProviderKeyCards = ({ onError }: { onError: (title: string, message: string) => void }) => {
  const { t } = useTranslation();
  const configs = useExternalProvidersSelector((snapshot) => snapshot.configs);
  const loadError = useExternalProvidersSelector((snapshot) => snapshot.error);
  const status = useExternalProvidersSelector((snapshot) => snapshot.status);

  useMountEffect(() => {
    ensureExternalProvidersLoaded().catch(() => {
      // The snapshot records the failure; rendered inline below.
    });
  });

  // Only a settled failure renders as an error — a retry in flight after a
  // remount shows nothing rather than the previous attempt's stale message.
  if (status === 'error') {
    return (
      <Text color="fg.error" fontSize="2xs">
        {t('models.externalProvidersUnavailable', { error: loadError ?? t('models.failedToLoadExternalProviders') })}
      </Text>
    );
  }

  // A server with no external providers is a normal state, not a blank one.
  // Nothing renders while loading — no flicker on the happy path.
  if (status === 'loaded' && (configs ?? []).length === 0) {
    return (
      <Text color="fg.subtle" fontSize="2xs">
        {t('models.noExternalProviders')}
      </Text>
    );
  }

  return (
    <>
      {(configs ?? []).map((config) => {
        const presentation = EXTERNAL_PROVIDER_PRESENTATION[config.provider_id] ?? {
          icon: HexagonIcon,
          title: config.provider_id,
        };
        const placeholder = presentation.placeholderKey
          ? t(presentation.placeholderKey)
          : (presentation.placeholder ?? t('models.apiKey'));

        return (
          <ExternalProviderKeyCard
            config={config}
            key={config.provider_id}
            description={t('models.externalProviderKeyDescription')}
            icon={presentation.icon}
            placeholder={placeholder}
            title={presentation.title}
            onError={(message) => onError(presentation.title, message)}
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
  placeholder,
  title,
}: {
  config: ExternalProviderConfig;
  description: string;
  icon: ElementType;
  onError: (message: string) => void;
  placeholder: string;
  title: string;
}) => {
  const { t } = useTranslation();
  const [apiKeyDraft, setApiKeyDraft] = useState('');
  const [baseUrlDraft, setBaseUrlDraft] = useState(config.base_url ?? '');
  const [overrideBaseUrl, setOverrideBaseUrl] = useState(config.base_url !== null);
  const { isBusy, run } = useScopedAction();

  const hasApiKeyDraft = apiKeyDraft.trim().length > 0;
  const hasBaseUrlChange = overrideBaseUrl ? baseUrlDraft.trim() !== (config.base_url ?? '') : config.base_url !== null;

  const runProviderAction = (
    action: () => Promise<ExternalProviderConfig>,
    onSuccess?: (next: ExternalProviderConfig, owner: AccountScope) => void
  ): Promise<boolean> =>
    run(
      async (owner) => {
        const nextConfig = await action();

        assertAccountScopeCurrent(owner);
        setApiKeyDraft('');
        setBaseUrlDraft(nextConfig.base_url ?? '');
        setOverrideBaseUrl(nextConfig.base_url !== null);
        onSuccess?.(nextConfig, owner);
      },
      (_message, error) => onError(error instanceof Error ? error.message : t('common.somethingWentWrong'))
    );

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

    await runProviderAction(
      () => saveExternalProviderConfig(config.provider_id, nextConfig),
      (next, owner) => {
        if (apiKeyWasSet && next.api_key_configured) {
          // The backend queues this provider's external starter models on
          // key-set; pull the new jobs and refresh starters so its rows flip to
          // Installed without an app restart.
          void refreshInstalls(owner);
          refreshStartersIfLoaded();
        }
      }
    );
  };

  const canClear = config.api_key_configured || config.base_url !== null;

  return (
    <KeyCardShell
      description={description}
      icon={icon}
      status={{
        label: config.api_key_configured ? t('models.keyStatus.configured') : t('models.keyStatus.unknown'),
        palette: config.api_key_configured ? 'green' : 'gray',
      }}
      title={title}
    >
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
            onClick={() => void runProviderAction(() => clearExternalProviderConfig(config.provider_id))}
          >
            {t('common.clear')}
          </Button>
        ) : null}
      </HStack>
    </KeyCardShell>
  );
};
