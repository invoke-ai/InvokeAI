import { HStack, Stack, Text } from '@chakra-ui/react';
import { promptTemplatesQueryOptions } from '@features/generation/queries';
import { systemPromptsQueryOptions } from '@features/generation/systemPrompts';
import { getFieldRecordId } from '@features/workflow/utility';
import { Combobox, IconButton, Tooltip } from '@platform/ui';
import { useQuery } from '@tanstack/react-query';
import { RotateCcwIcon, XIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowFieldInputProps } from './WorkflowFieldInput';

interface RecordPickerLabels {
  clear: string;
  missing: string;
  noMatches: string;
  search: string;
}

interface RecordListQuery {
  isError: boolean;
  isFetching: boolean;
  records: readonly { id: string; name: string }[] | undefined;
  refetch: () => void;
}

/**
 * Picks one server record by id for `{ [idProp]: id }` values. A stored id the list no longer holds stays put and
 * is shown as missing: readiness cannot see the list, so the backend reports the stale reference on invoke.
 */
const RecordPickerInput = ({
  idProp,
  invalid,
  labels,
  onChange,
  query,
  template,
  value,
}: WorkflowFieldInputProps & { idProp: string; labels: RecordPickerLabels; query: RecordListQuery }) => {
  const { t } = useTranslation();
  const { isError, isFetching, records, refetch } = query;
  const messageId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const selectedId = getFieldRecordId(value, idProp);
  const isMissing = selectedId !== null && records !== undefined && !records.some((record) => record.id === selectedId);
  const message = isMissing ? labels.missing : isError ? t('nodes.recordListFailed') : null;
  const options = useMemo(() => {
    const base = (records ?? []).map((record) => ({ label: record.name, value: record.id }));

    return isMissing && selectedId !== null ? [{ label: labels.missing, value: selectedId }, ...base] : base;
  }, [isMissing, labels.missing, records, selectedId]);
  const inputProps = useMemo(
    () => ({ ref: inputRef, 'aria-describedby': message !== null ? messageId : undefined }),
    [message, messageId]
  );
  const onPick = useCallback((id: string) => onChange({ [idProp]: id }), [idProp, onChange]);
  // The clear button leaves with the value; hand keyboard focus to the input before it goes.
  const onClear = useCallback(() => {
    inputRef.current?.focus();
    onChange(undefined);
  }, [onChange]);
  const onRetry = useCallback(() => refetch(), [refetch]);

  return (
    <Stack gap="1" minW="0" w="full">
      <HStack gap="1" minW="0" w="full">
        <Combobox
          aria-label={template.title}
          className="nodrag nowheel"
          flex="1"
          inputProps={inputProps}
          invalid={invalid || isMissing}
          noResultsText={labels.noMatches}
          options={options}
          searchPlaceholder={isFetching && records === undefined ? t('nodes.recordListLoading') : labels.search}
          value={selectedId}
          onValueChange={onPick}
        />
        {isError ? (
          <Tooltip content={t('common.retry')}>
            <IconButton
              aria-label={t('common.retry')}
              className="nodrag"
              disabled={isFetching}
              size="xs"
              variant="ghost"
              onClick={onRetry}
            >
              <RotateCcwIcon />
            </IconButton>
          </Tooltip>
        ) : null}
        {selectedId !== null ? (
          <Tooltip content={labels.clear}>
            <IconButton aria-label={labels.clear} className="nodrag" size="xs" variant="ghost" onClick={onClear}>
              <XIcon />
            </IconButton>
          </Tooltip>
        ) : null}
      </HStack>
      {message !== null ? (
        <Text color="fg.error" fontSize="2xs" id={messageId}>
          {message}
        </Text>
      ) : null}
    </Stack>
  );
};

export const StylePresetInput = (props: WorkflowFieldInputProps) => {
  const { t } = useTranslation();
  const { data, isError, isFetching, refetch } = useQuery(promptTemplatesQueryOptions());
  const labels = useMemo<RecordPickerLabels>(
    () => ({
      clear: t('nodes.stylePresetClear'),
      missing: t('nodes.stylePresetMissing'),
      noMatches: t('nodes.noMatchingStylePresets'),
      search: t('nodes.stylePresetSearch'),
    }),
    [t]
  );
  const query = useMemo<RecordListQuery>(
    () => ({ isError, isFetching, records: data, refetch: () => void refetch() }),
    [data, isError, isFetching, refetch]
  );

  return <RecordPickerInput {...props} idProp="style_preset_id" labels={labels} query={query} />;
};

export const SystemPromptInput = (props: WorkflowFieldInputProps) => {
  const { t } = useTranslation();
  const { data, isError, isFetching, refetch } = useQuery(systemPromptsQueryOptions());
  const labels = useMemo<RecordPickerLabels>(
    () => ({
      clear: t('nodes.systemPromptClear'),
      missing: t('nodes.systemPromptMissing'),
      noMatches: t('nodes.noMatchingSystemPrompts'),
      search: t('nodes.systemPromptSearch'),
    }),
    [t]
  );
  const query = useMemo<RecordListQuery>(
    () => ({ isError, isFetching, records: data, refetch: () => void refetch() }),
    [data, isError, isFetching, refetch]
  );

  return <RecordPickerInput {...props} idProp="system_prompt_id" labels={labels} query={query} />;
};
