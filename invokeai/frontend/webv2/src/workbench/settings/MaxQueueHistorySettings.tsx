import type { ChangeEvent, FormEvent } from 'react';

import { HStack, Input, Spinner, Stack, Text } from '@chakra-ui/react';
import { useCapabilities } from '@features/identity';
import { refreshGenerationDevices, updateMaxQueueHistory, useGenerationDevices } from '@features/queue/devices';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { ModifiedSettingIndicator } from '@platform/ui/settings/ModifiedSettingIndicator';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const LIMIT_PATTERN = /^\d+$/;

/** Blank keeps every finished item; 0 prunes all of them; a count keeps the newest N. */
const parseLimit = (text: string): number | null | 'invalid' => {
  const trimmed = text.trim();

  if (trimmed === '') {
    return null;
  }

  return LIMIT_PATTERN.test(trimmed) && Number.isSafeInteger(Number(trimmed)) ? Number(trimmed) : 'invalid';
};

const formatLimit = (value: number | null | undefined): string => (typeof value === 'number' ? String(value) : '');

/** `max_queue_history` is admin-only server configuration applied at the next startup. */
export const MaxQueueHistorySettings = () => {
  const { t } = useTranslation();
  const { canManageAppConfig } = useCapabilities();
  const { loadState, maxQueueHistory } = useGenerationDevices();
  const [draft, setDraft] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [didChange, setDidChange] = useState(false);

  useMountEffect(() => {
    void refreshGenerationDevices();
  });

  const [retrying, setRetrying] = useState(false);
  // A successful retry replaces the focused Retry button with the field; focus follows onto it.
  const focusAfterRetryRef = useRef(false);
  const onRetry = useCallback(async () => {
    focusAfterRetryRef.current = true;
    setRetrying(true);
    await refreshGenerationDevices();
    setRetrying(false);
  }, []);
  const focusRestoredInput = useCallback((node: HTMLInputElement | null) => {
    if (node && focusAfterRetryRef.current) {
      focusAfterRetryRef.current = false;
      node.focus();
    }
  }, []);

  const savedText = formatLimit(maxQueueHistory);
  const text = draft ?? savedText;
  const parsed = parseLimit(text);
  const isDirty = draft !== null && draft.trim() !== savedText;

  const onChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setDraft(event.target.value);
    setSaveError(null);
  }, []);

  const onSubmit = useCallback(
    async (event: FormEvent) => {
      event.preventDefault();
      if (parsed === 'invalid' || !isDirty) {
        return;
      }

      setSaving(true);
      setSaveError(null);
      try {
        await updateMaxQueueHistory(parsed);
        setDraft(null);
        setDidChange(true);
      } catch (failure) {
        setSaveError(failure instanceof Error ? failure.message : t('settings.maxQueueHistory.saveFailed'));
      } finally {
        setSaving(false);
      }
    },
    [isDirty, parsed, t]
  );

  // A retry keeps its own button (showing progress) mounted, so focus stays on it if the read fails again.
  if ((loadState === 'loading' || loadState === 'idle') && !retrying) {
    return <Spinner size="lg" />;
  }

  if (!canManageAppConfig) {
    return (
      <Text color="fg.muted" fontSize="md">
        {t('settings.maxQueueHistory.adminOnly')}
      </Text>
    );
  }

  // A save stays acknowledged even when the server's answer to it cannot be read back.
  const restartNotice = didChange ? (
    <Text color="fg.muted" fontSize="md">
      {t('settings.maxQueueHistory.restartRequired')}
    </Text>
  ) : null;

  // An unknown limit is never shown as the blank "keep all" field: saving from it would overwrite the real limit.
  if (maxQueueHistory === undefined) {
    return (
      <Stack gap="2">
        <HStack gap="2">
          <Text color="fg.error" fontSize="md" role="alert">
            {t('settings.maxQueueHistory.readFailed')}
          </Text>
          <Button loading={retrying} size="sm" variant="surface" onClick={onRetry}>
            {t('common.retry')}
          </Button>
        </HStack>
        {restartNotice}
      </Stack>
    );
  }

  return (
    <Stack as="form" gap="2" onSubmit={onSubmit}>
      <Text color="fg.muted" fontSize="md">
        {t('settings.maxQueueHistory.description')}
      </Text>
      <HStack gap="2">
        <Input
          ref={focusRestoredInput}
          aria-invalid={parsed === 'invalid' ? true : undefined}
          aria-label={t('settings.maxQueueHistory.inputLabel')}
          inputMode="numeric"
          maxW="10rem"
          placeholder={t('settings.maxQueueHistory.keepAll')}
          size="sm"
          value={text}
          onChange={onChange}
        />
        <Button disabled={!isDirty || parsed === 'invalid'} loading={saving} size="sm" type="submit" variant="surface">
          {t('common.save')}
        </Button>
        {typeof maxQueueHistory === 'number' ? (
          <ModifiedSettingIndicator label={t('settings.maxQueueHistory.modified')} />
        ) : null}
      </HStack>
      {parsed === 'invalid' ? (
        <Text color="fg.error" fontSize="md">
          {t('settings.maxQueueHistory.invalid')}
        </Text>
      ) : null}
      {saveError ? (
        <Text color="fg.error" fontSize="md" role="alert">
          {saveError}
        </Text>
      ) : null}
      {restartNotice}
    </Stack>
  );
};
