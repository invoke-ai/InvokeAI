import type { ChangeEvent, FormEvent } from 'react';

import { HStack, Input, Spinner, Stack, Text } from '@chakra-ui/react';
import { useCapabilities } from '@features/identity';
import { refreshGenerationDevices, updateMaxQueueHistory, useGenerationDevices } from '@features/queue/devices';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { ModifiedSettingIndicator } from '@platform/ui/settings/ModifiedSettingIndicator';
import { useCallback, useState } from 'react';

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
  const { canManageAppConfig } = useCapabilities();
  const { error, loadState, maxQueueHistory } = useGenerationDevices();
  const [draft, setDraft] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [didChange, setDidChange] = useState(false);

  useMountEffect(() => {
    void refreshGenerationDevices();
  });

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
        setSaveError(failure instanceof Error ? failure.message : 'Could not save the queue history limit.');
      } finally {
        setSaving(false);
      }
    },
    [isDirty, parsed]
  );

  if (loadState === 'loading' || loadState === 'idle') {
    return <Spinner size="lg" />;
  }

  if (error) {
    return (
      <Text color="fg.error" fontSize="md">
        {error}
      </Text>
    );
  }

  if (!canManageAppConfig) {
    return (
      <Text color="fg.muted" fontSize="md">
        Only an administrator can change how much queue history the server keeps.
      </Text>
    );
  }

  return (
    <Stack as="form" gap="2" onSubmit={onSubmit}>
      <Text color="fg.muted" fontSize="md">
        Finished, failed and canceled items kept when InvokeAI starts. Leave blank to keep everything, or enter 0 to
        clear finished history on every start.
      </Text>
      <HStack gap="2">
        <Input
          aria-invalid={parsed === 'invalid' ? true : undefined}
          aria-label="Queue history limit"
          inputMode="numeric"
          maxW="10rem"
          placeholder="Keep all"
          size="sm"
          value={text}
          onChange={onChange}
        />
        <Button disabled={!isDirty || parsed === 'invalid'} loading={saving} size="sm" type="submit" variant="surface">
          Save
        </Button>
        {typeof maxQueueHistory === 'number' ? <ModifiedSettingIndicator label="Queue history" /> : null}
      </HStack>
      {parsed === 'invalid' ? (
        <Text color="fg.error" fontSize="md">
          Enter a whole number of items, or leave the field blank.
        </Text>
      ) : null}
      {saveError ? (
        <Text color="fg.error" fontSize="md">
          {saveError}
        </Text>
      ) : null}
      {didChange ? (
        <Text color="fg.muted" fontSize="md">
          Restart InvokeAI for changes to take effect.
        </Text>
      ) : null}
    </Stack>
  );
};
