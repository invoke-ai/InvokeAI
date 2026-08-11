/* eslint-disable react/react-compiler, react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { HFTokenStatus } from '@features/models/core/types';
import type { ElementType } from 'react';

import { Grid, HStack, Input, Stack, Text } from '@chakra-ui/react';
import { getHFTokenStatus, resetHFToken, setHFToken } from '@features/models/data/api';
import { clearCivitaiApiKey, getCivitaiApiKey, setCivitaiApiKey } from '@features/models/data/apiKeys';
import { ExternalProviderKeyCards } from '@features/models/ui/credentials/ExternalProviderKeyCards';
import { KeyCardShell, type KeyStatusBadge } from '@features/models/ui/credentials/KeyCardShell';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useScopedAction } from '@platform/react/useScopedAction';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui';
import { HexagonIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SiHuggingface } from 'react-icons/si';

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
  onClear?: (owner: AccountScope) => Promise<void> | void;
  /** Save/clear failures (network, backend rejection) land here. */
  onError: (message: string) => void;
  onSave: (key: string, owner: AccountScope) => Promise<void> | void;
  /** Hint at the expected shape, e.g. "hf_…" or "sk-…". */
  placeholder: string;
  status: KeyStatusBadge | null;
  title: string;
}) => {
  const { t } = useTranslation();
  const [draft, setDraft] = useState('');
  const { isBusy, run } = useScopedAction();

  const handleActionError = (_message: string, error: unknown) =>
    onError(getApiErrorMessage(error, t('common.somethingWentWrong')));

  const handleSave = () => {
    const key = draft.trim();

    if (key.length === 0) {
      return;
    }

    // Keep the draft on failure so the key can be corrected and retried.
    void run(async (owner) => {
      await onSave(key, owner);
      assertAccountScopeCurrent(owner);
      setDraft('');
    }, handleActionError);
  };

  return (
    <KeyCardShell description={description} icon={icon} status={status} title={title}>
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
              handleSave();
            }
          }}
        />
        <Button
          disabled={draft.trim().length === 0 || isLoading}
          loading={isBusy && draft.trim().length > 0}
          size="xs"
          variant="solid"
          onClick={handleSave}
        >
          {t('common.save')}
        </Button>
        {onClear ? (
          <Button
            disabled={isBusy || isLoading}
            size="sm"
            variant="ghost"
            onClick={() =>
              void run(async (owner) => {
                await onClear(owner);
              }, handleActionError)
            }
          >
            {t('common.clear')}
          </Button>
        ) : null}
      </HStack>
    </KeyCardShell>
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

  useMountEffect(() => {
    const owner = captureAccountScope();
    let isMounted = true;

    getHFTokenStatus(owner.signal)
      .then((tokenStatus) => {
        if (isMounted && isAccountScopeCurrent(owner)) {
          setStatus(tokenStatus);
        }
      })
      .catch(() => {
        if (isMounted && isAccountScopeCurrent(owner)) {
          setStatus('unknown');
        }
      });

    return () => {
      isMounted = false;
    };
  });

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
          ? async (owner) => {
              const nextStatus = await resetHFToken(owner.signal);

              assertAccountScopeCurrent(owner);
              setStatus(nextStatus);
            }
          : undefined
      }
      onSave={async (key, owner) => {
        const nextStatus = await setHFToken(key, owner.signal);

        assertAccountScopeCurrent(owner);
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
