import type {
  WildcardImportEntry,
  WildcardImportRejection,
  WildcardImportResolution,
} from '@features/generation/core/wildcardTransfer';

import { HStack, SegmentGroup, Stack, Text } from '@chakra-ui/react';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Scrollable } from '@platform/ui/Scrollable';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const RESOLUTIONS: WildcardImportResolution[] = ['skip', 'replace', 'keepBoth'];

const RESOLUTION_LABEL_KEY: Record<WildcardImportResolution, string> = {
  keepBoth: 'widgets.generate.dynamicPrompts.importKeepBoth',
  replace: 'widgets.generate.dynamicPrompts.importReplace',
  skip: 'widgets.generate.dynamicPrompts.importSkip',
};

const REJECTION_KEY: Record<WildcardImportRejection, string> = {
  duplicate: 'widgets.generate.dynamicPrompts.importRejectedDuplicate',
  invalid: 'widgets.generate.dynamicPrompts.importRejectedInvalid',
  noValues: 'widgets.generate.dynamicPrompts.importRejectedNoValues',
  tooLong: 'widgets.generate.dynamicPrompts.importRejectedTooLong',
  tooManyValues: 'widgets.generate.dynamicPrompts.importRejectedTooManyValues',
  valueTooLong: 'widgets.generate.dynamicPrompts.importRejectedValueTooLong',
};

/** Default conflicts to skip and show explicit rejection reasons. */
export const WildcardImportDialog = ({
  entries,
  onCancel,
  onConfirm,
}: {
  entries: readonly WildcardImportEntry[];
  onCancel: () => void;
  onConfirm: (resolutions: Record<string, WildcardImportResolution>) => Promise<void>;
}) => {
  const { t } = useTranslation();
  const [resolutions, setResolutions] = useState<Record<string, WildcardImportResolution>>({});
  // Control set-all state so it cannot accidentally overwrite later per-row choices.
  const [allResolution, setAllResolution] = useState<WildcardImportResolution | null>(null);

  const conflicts = useMemo(
    () => entries.filter((entry) => entry.rejection === null && entry.conflictId !== null),
    [entries]
  );
  const freshCount = useMemo(
    () => entries.filter((entry) => entry.rejection === null && entry.conflictId === null).length,
    [entries]
  );
  const rejected = useMemo(
    () => entries.flatMap((entry) => (entry.rejection ? [{ name: entry.name, rejection: entry.rejection }] : [])),
    [entries]
  );

  const setResolution = useCallback((name: string, resolution: WildcardImportResolution) => {
    // A single row disagreeing means the set-all row no longer describes them.
    setAllResolution(null);
    setResolutions((current) => ({ ...current, [name]: resolution }));
  }, []);

  const setAllResolutions = useCallback(
    (resolution: WildcardImportResolution) => {
      setAllResolution(resolution);
      setResolutions(Object.fromEntries(conflicts.map((entry) => [entry.name, resolution])));
    },
    [conflicts]
  );

  const handleConfirm = useCallback(() => onConfirm(resolutions), [onConfirm, resolutions]);

  // The body changes on every resolution edit.
  // oxlint-disable-next-line react-perf/jsx-no-jsx-as-prop
  const body = (
    <Stack gap="3">
      <Text color="fg.subtle" fontSize="xs">
        {t('widgets.generate.dynamicPrompts.importSummary', { conflicts: conflicts.length, fresh: freshCount })}
      </Text>

      {conflicts.length > 0 ? (
        <Stack gap="2">
          <HStack justify="space-between">
            <Text fontSize="xs" fontWeight="600">
              {t('widgets.generate.dynamicPrompts.importAlreadyExist')}
            </Text>
            {/* One decision for a folder of forty; the rows below still win. */}
            {conflicts.length > 1 ? <ResolutionControl value={allResolution} onChange={setAllResolutions} /> : null}
          </HStack>
          <Scrollable maxH="12rem" label={t('widgets.generate.dynamicPrompts.importAlreadyExist')}>
            <Stack gap="1" pr="1">
              {conflicts.map((entry) => (
                <ConflictRow
                  key={entry.name}
                  name={entry.name}
                  resolution={resolutions[entry.name] ?? 'skip'}
                  valueCount={entry.values.length}
                  onChange={setResolution}
                />
              ))}
            </Stack>
          </Scrollable>
        </Stack>
      ) : null}

      {rejected.length > 0 ? (
        <Stack gap="1">
          <Text fontSize="xs" fontWeight="600">
            {t('widgets.generate.dynamicPrompts.importCannotImport')}
          </Text>
          {/* Bound rejected-file lists so dialog actions remain reachable. */}
          <Scrollable maxH="8rem" label={t('widgets.generate.dynamicPrompts.importCannotImport')}>
            <Stack gap="1" pr="1">
              {rejected.map((entry, index) => (
                <HStack justify="space-between" key={`${entry.name}-${index}`}>
                  <MiddleTruncate
                    fontFamily="mono"
                    fontSize="2xs"
                    text={entry.name || t('widgets.generate.dynamicPrompts.importUnnamed')}
                  />
                  <Text color="fg.subtle" fontSize="2xs" flexShrink="0">
                    {t(REJECTION_KEY[entry.rejection])}
                  </Text>
                </HStack>
              ))}
            </Stack>
          </Scrollable>
        </Stack>
      ) : null}
    </Stack>
  );

  return (
    <ConfirmDialog
      body={body}
      confirmLabel={t('widgets.generate.dynamicPrompts.import')}
      isDestructive={false}
      isOpen
      title={t('widgets.generate.dynamicPrompts.importWildcards')}
      onClose={onCancel}
      onConfirm={handleConfirm}
    />
  );
};

const ResolutionControl = ({
  onChange,
  value,
}: {
  onChange: (resolution: WildcardImportResolution) => void;
  value?: WildcardImportResolution | null;
}) => {
  const { t } = useTranslation();
  const handleValueChange = useCallback(
    (event: { value: string | null }) => {
      if (event.value) {
        onChange(event.value as WildcardImportResolution);
      }
    },
    [onChange]
  );

  return (
    <SegmentGroup.Root size="xs" value={value ?? null} onValueChange={handleValueChange}>
      <SegmentGroup.Indicator />
      {RESOLUTIONS.map((resolution) => (
        <SegmentGroup.Item key={resolution} value={resolution}>
          <SegmentGroup.ItemHiddenInput />
          <SegmentGroup.ItemText>{t(RESOLUTION_LABEL_KEY[resolution])}</SegmentGroup.ItemText>
        </SegmentGroup.Item>
      ))}
    </SegmentGroup.Root>
  );
};

const ConflictRow = ({
  name,
  onChange,
  resolution,
  valueCount,
}: {
  name: string;
  resolution: WildcardImportResolution;
  valueCount: number;
  onChange: (name: string, resolution: WildcardImportResolution) => void;
}) => {
  const { t } = useTranslation();
  const handleChange = useCallback((next: WildcardImportResolution) => onChange(name, next), [name, onChange]);

  return (
    <HStack gap="2" justify="space-between">
      <Stack gap="0" minW="0">
        <MiddleTruncate fontFamily="mono" fontSize="2xs" text={`__${name}__`} />
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.generate.dynamicPrompts.importValueCount', { count: valueCount })}
        </Text>
      </Stack>
      <ResolutionControl value={resolution} onChange={handleChange} />
    </HStack>
  );
};
