import type { HFTokenStatus } from '@workbench/models/types';

import { Badge, Flex, Grid, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui/Button';
import {
  getExternalProviderConfigs,
  getHFTokenStatus,
  resetExternalProviderConfig,
  resetHFToken,
  setExternalProviderConfig,
  setHFToken,
  type ExternalProviderConfig,
} from '@workbench/models/api';
import { clearCivitaiApiKey, getCivitaiApiKey, setCivitaiApiKey } from '@workbench/models/apiKeys';
import { useNotify } from '@workbench/useNotify';
import { HexagonIcon } from 'lucide-react';
import { useEffect, useState, type ElementType } from 'react';
import { SiAlibabacloud, SiBytedance, SiGooglegemini, SiHuggingface, SiOpenai } from 'react-icons/si';

/**
 * Credentials for every model source, as one uniform grid of key cards:
 * HuggingFace (server-stored, verified), Civitai (browser-local), and the
 * external image providers (server config). Each card shares the same
 * edit/display anatomy: icon, status, prefix-hinted input, save/clear.
 */
export const ApiKeysSection = () => {
  const notify = useNotify();

  return (
    <Stack gap="3">
      <Text color="fg.subtle" fontSize="2xs">
        Keys are used when installing protected models and when generating with external API models.
      </Text>
      <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(19rem, 1fr))">
        <HuggingFaceKeyCard onError={notify.error} />
        <CivitaiKeyCard onError={(message) => notify.error('Civitai key', message)} />
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
  const [draft, setDraft] = useState('');
  const [isBusy, setIsBusy] = useState(false);

  const runAction = async (action: () => Promise<void> | void): Promise<boolean> => {
    setIsBusy(true);

    try {
      await action();

      return true;
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Something went wrong.');

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
          aria-label={`${title} API key`}
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
          Save
        </Button>
        {onClear ? (
          <Button disabled={isBusy || isLoading} size="sm" variant="ghost" onClick={() => void runAction(onClear)}>
            Clear
          </Button>
        ) : null}
      </HStack>
    </Stack>
  );
};

const HF_STATUS_BADGES: Record<HFTokenStatus, KeyStatusBadge> = {
  invalid: { label: 'Invalid', palette: 'red' },
  unknown: { label: 'Not set', palette: 'gray' },
  valid: { label: 'Valid', palette: 'green' },
};

const HuggingFaceKeyCard = ({ onError }: { onError: (title: string, message: string) => void }) => {
  const [status, setStatus] = useState<HFTokenStatus | null>(null);

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
  }, []);

  return (
    <ApiKeyCard
      description="For gated repos like FLUX. Stored on the InvokeAI server and verified with HuggingFace."
      icon={SiHuggingface}
      isLoading={status === null}
      placeholder="hf_…"
      status={status ? HF_STATUS_BADGES[status] : null}
      title="HuggingFace"
      onError={(message) => onError('HuggingFace token', message)}
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
          onError('HuggingFace token', 'HuggingFace rejected this token. Check that it has read access.');
        }
      }}
    />
  );
};

const CivitaiKeyCard = ({ onError }: { onError: (message: string) => void }) => {
  const [hasKey, setHasKey] = useState(() => getCivitaiApiKey() !== null);

  return (
    <ApiKeyCard
      description="Attached automatically when installing civitai.com models that require login. Stored in this browser only."
      icon={HexagonIcon}
      placeholder="32-character key from civitai.com/user/account"
      status={hasKey ? { label: 'Saved', palette: 'green' } : { label: 'Not set', palette: 'gray' }}
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
  seedream: { icon: SiBytedance, placeholder: 'API key from BytePlus console', title: 'Seedream' },
};

const ExternalProviderKeyCards = ({ onError }: { onError: (title: string, message: string) => void }) => {
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
          setLoadError(error instanceof Error ? error.message : 'Failed to load external providers.');
        }
      });

    return () => {
      isStale = true;
    };
  }, []);

  const replaceConfig = (next: ExternalProviderConfig) => {
    setConfigs((current) => (current ?? []).map((config) => (config.provider_id === next.provider_id ? next : config)));
  };

  if (loadError) {
    return (
      <Text color="fg.error" fontSize="2xs">
        External providers unavailable: {loadError}
      </Text>
    );
  }

  return (
    <>
      {(configs ?? []).map((config) => {
        const presentation = EXTERNAL_PROVIDER_PRESENTATION[config.provider_id] ?? {
          icon: HexagonIcon,
          placeholder: 'API key',
          title: config.provider_id,
        };

        return (
          <ApiKeyCard
            key={config.provider_id}
            description="External image generator. The key is stored in the InvokeAI server config and used by external API models."
            icon={presentation.icon}
            placeholder={presentation.placeholder}
            status={
              config.api_key_configured
                ? { label: 'Configured', palette: 'green' }
                : { label: 'Not set', palette: 'gray' }
            }
            title={presentation.title}
            onError={(message) => onError(presentation.title, message)}
            onClear={
              config.api_key_configured
                ? async () => {
                    replaceConfig(await resetExternalProviderConfig(config.provider_id));
                  }
                : undefined
            }
            onSave={async (key) => {
              replaceConfig(await setExternalProviderConfig(config.provider_id, { api_key: key }));
            }}
          />
        );
      })}
    </>
  );
};
