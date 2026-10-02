import type { ChangeEvent } from 'react';

import { Box, chakra, Checkbox, createListCollection, Field, HStack, Input, Stack, Textarea } from '@chakra-ui/react';
import { galleryBoardsOptions } from '@features/gallery/queries';
import { getWorkflowGeneratorQueryOptions, type WorkflowGeneratorQueryResult } from '@features/workflow/generators';
import {
  getDefaultWorkflowGeneratorValue,
  getWorkflowGeneratorInvalidReason,
  getWorkflowGeneratorPropertyInvalidReason,
  getWorkflowGeneratorRequestedCount,
  getWorkflowGeneratorVariantDefaults,
  isRandomWorkflowGeneratorValue,
  isWorkflowGeneratorVariant,
  WORKFLOW_BATCH_MAX_ITEMS,
  WORKFLOW_DYNAMIC_PROMPTS_MAX,
  parseWorkflowGeneratorValue,
  resolveWorkflowGeneratorValue,
  WORKFLOW_GENERATOR_VARIANTS,
  type WorkflowAsyncGeneratorRequest,
  type WorkflowGeneratorValue,
  type WorkflowGeneratorVariant,
} from '@features/workflow/utility';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, Combobox, Select, toaster } from '@platform/ui';
import { useQuery, type UseQueryOptions } from '@tanstack/react-query';
import { UploadIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowFieldInputProps } from './WorkflowFieldInput';

import { NumericInput, type NumericInputTemplate } from './NumericInput';

const VARIANT_LABEL_KEYS: Record<WorkflowGeneratorVariant, string> = {
  float_generator_arithmetic_sequence: 'nodes.arithmeticSequence',
  float_generator_linear_distribution: 'nodes.linearDistribution',
  float_generator_parse_string: 'nodes.parseString',
  float_generator_random_distribution_uniform: 'nodes.uniformRandomDistribution',
  image_generator_images_from_board: 'nodes.generatorImagesFromBoard',
  integer_generator_arithmetic_sequence: 'nodes.arithmeticSequence',
  integer_generator_linear_distribution: 'nodes.linearDistribution',
  integer_generator_parse_string: 'nodes.parseString',
  integer_generator_random_distribution_uniform: 'nodes.uniformRandomDistribution',
  string_generator_dynamic_prompts_combinatorial: 'nodes.dynamicPromptsCombinatorial',
  string_generator_dynamic_prompts_random: 'nodes.dynamicPromptsRandom',
  string_generator_parse_string: 'nodes.parseString',
};

const FILE_SIZE_LIMIT = 128 * 1024;
const PREVIEW_ITEM_LIMIT = 200;
const PREVIEW_DEBOUNCE_MS = 300;
const SELECT_VALUE_TEXT_PROPS = { placeholder: 'Select…' };
const HIDDEN_FILE_INPUT_STYLE = { display: 'none' } as const;
const CATEGORY_OPTIONS = [
  { labelKey: 'nodes.generatorCategoryImages', value: 'images' },
  { labelKey: 'nodes.generatorCategoryAssets', value: 'assets' },
] as const;

/**
 * One labelled number in a settings row, on the shared numeric control: committed as typed, `null` when cleared,
 * and marked by the generator's own rule for that setting rather than corrected here.
 */
const NumberSetting = ({
  ariaLabel,
  disabled,
  id,
  integer,
  invalid,
  label,
  max,
  min,
  value,
  onCommit,
}: {
  /** The accessible name when the visible label lives elsewhere (the seed row's checkbox). */
  ariaLabel?: string;
  disabled?: boolean;
  id: string;
  integer: boolean;
  invalid: boolean;
  label: string;
  max?: number;
  min?: number;
  value: number | null;
  onCommit: (value: number | null) => void;
}) => {
  const template = useMemo<NumericInputTemplate>(
    () => ({
      exclusiveMaximum: null,
      exclusiveMinimum: null,
      maximum: max ?? null,
      minimum: min ?? null,
      multipleOf: null,
      title: label || ariaLabel || '',
      type: { batch: false, cardinality: 'SINGLE', name: integer ? 'IntegerField' : 'FloatField' },
    }),
    [ariaLabel, integer, label, max, min]
  );
  const onChange = useCallback((next: number | undefined) => onCommit(next ?? null), [onCommit]);

  return (
    <HStack flex="1" gap="1" minW="0">
      {label ? (
        <chakra.label color="fg.subtle" flexShrink={0} fontSize="2xs" htmlFor={`${id}-number-input`} minW="10">
          {label}
        </chakra.label>
      ) : null}
      {/* The host's Field.Root marks every control inside it invalid; each setting scopes its own validity. */}
      <Field.Root flex="1" invalid={invalid} minW="12">
        <NumericInput
          ariaLabel={ariaLabel}
          disabled={disabled}
          id={id}
          invalid={invalid}
          size="2xs"
          template={template}
          value={value ?? undefined}
          onChange={onChange}
        />
      </Field.Root>
    </HStack>
  );
};

/** The seed row: unchecked draws fresh values every time, checked pins them to a seed. */
const SeedSetting = ({
  id,
  invalid,
  seed,
  onCommit,
}: {
  id: string;
  invalid: boolean;
  seed: number | null;
  onCommit: (seed: number | null) => void;
}) => {
  const { t } = useTranslation();
  const onCheckedChange = useCallback(
    (event: { checked: boolean | 'indeterminate' }) => onCommit(event.checked === true ? 0 : null),
    [onCommit]
  );
  // `null` already means "unseeded" here, so a cleared input keeps the pinned seed instead of unchecking the box.
  const onSeedCommit = useCallback(
    (next: number | null) => {
      if (next !== null) {
        onCommit(next);
      }
    },
    [onCommit]
  );

  return (
    <HStack gap="2" minW="0">
      {/* Ark controls take the enclosing field's control id; a field of their own keeps ids unique in the widget. */}
      <Field.Root flexShrink={0} gap="0" w="auto">
        <Checkbox.Root checked={seed !== null} colorPalette="accent" size="xs" onCheckedChange={onCheckedChange}>
          <Checkbox.HiddenInput />
          <Checkbox.Control />
          <Checkbox.Label fontSize="2xs">{t('nodes.generatorSeed')}</Checkbox.Label>
        </Checkbox.Root>
      </Field.Root>
      <NumberSetting
        ariaLabel={t('nodes.generatorSeed')}
        disabled={seed === null}
        id={`${id}-seed`}
        integer
        invalid={invalid}
        label=""
        min={0}
        value={seed ?? 0}
        onCommit={onSeedCommit}
      />
    </HStack>
  );
};

/** The multi-line source of a parse-string or dynamic-prompt generator, with the legacy load-from-file action. */
const InputSetting = ({ id, value, onCommit }: { id: string; value: string; onCommit: (value: string) => void }) => {
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const onTextChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => onCommit(event.currentTarget.value),
    [onCommit]
  );
  const onLoadClick = useCallback(() => fileInputRef.current?.click(), []);
  const onFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      event.currentTarget.value = '';

      if (!file) {
        return;
      }

      if (file.size > FILE_SIZE_LIMIT) {
        toaster.create({ title: t('nodes.generatorFileTooLarge'), type: 'error' });
        return;
      }

      void file.text().then(onCommit);
    },
    [onCommit, t]
  );

  return (
    <Stack gap="1">
      <HStack justify="space-between">
        <chakra.label color="fg.subtle" fontSize="2xs" htmlFor={id}>
          {t('nodes.generatorInput')}
        </chakra.label>
        <Button className="nodrag" size="2xs" variant="ghost" onClick={onLoadClick}>
          <UploadIcon />
          {t('nodes.generatorLoadFromFile')}
        </Button>
        <input
          ref={fileInputRef}
          accept=".txt,.csv,text/plain,text/csv"
          aria-label={t('nodes.generatorLoadFromFile')}
          style={HIDDEN_FILE_INPUT_STYLE}
          type="file"
          onChange={onFileChange}
        />
      </HStack>
      <Textarea
        className="nodrag nowheel"
        fontFamily="mono"
        id={id}
        resize="none"
        rows={4}
        size="xs"
        value={value}
        onChange={onTextChange}
      />
    </Stack>
  );
};

const BoardSetting = ({
  boardId,
  id,
  onCommit,
}: {
  boardId: string | undefined;
  id: string;
  onCommit: (boardId: string | undefined) => void;
}) => {
  const { t } = useTranslation();
  const boards = useQuery(galleryBoardsOptions());
  const isMissing =
    boardId !== undefined && boards.data !== undefined && !boards.data.some((board) => board.id === boardId);
  const options = useMemo(() => {
    const base = (boards.data ?? []).map((board) => ({ label: board.name, value: board.id }));

    return isMissing && boardId !== undefined
      ? [{ label: t('nodes.generatorBoardMissing'), value: boardId }, ...base]
      : base;
  }, [boardId, boards.data, isMissing, t]);
  const onValueChange = useCallback((value: string) => onCommit(value), [onCommit]);

  return (
    <Field.Root gap="0" invalid={isMissing} minW="0" w="full">
      <Combobox
        aria-label={t('nodes.generatorBoard')}
        className="nodrag nowheel"
        id={id}
        invalid={isMissing}
        noResultsText={t('nodes.generatorSelectBoard')}
        options={options}
        searchPlaceholder={boards.isLoading ? t('nodes.recordListLoading') : t('nodes.generatorSelectBoard')}
        value={boardId ?? null}
        onValueChange={onValueChange}
      />
    </Field.Root>
  );
};

const formatItem = (item: unknown): string =>
  typeof item === 'number' ? String(Math.round(item * 100) / 100) : typeof item === 'string' ? item : '';

const PreviewBox = ({ children, tone = 'fg.muted' }: { children: string; tone?: string }) => (
  <Box
    bg="bg.subtle"
    className="nowheel"
    color={tone}
    fontFamily="mono"
    fontSize="2xs"
    maxH="32"
    overflowY="auto"
    px="1.5"
    py="1"
    rounded="sm"
    whiteSpace="pre-wrap"
    wordBreak="break-word"
  >
    {children}
  </Box>
);

const AsyncPreview = ({ request }: { request: WorkflowAsyncGeneratorRequest }) => {
  const { t } = useTranslation();
  // Both request kinds select into the same result shape; the union of option types is wider than useQuery wants.
  // Query identity comes from the key hash, so a fresh options object per render costs nothing.
  const query = useQuery(
    getWorkflowGeneratorQueryOptions(request) as unknown as UseQueryOptions<
      unknown,
      Error,
      WorkflowGeneratorQueryResult
    >
  );

  if (query.isError) {
    return <PreviewBox tone="fg.error">{t('nodes.generatorPreviewFailed')}</PreviewBox>;
  }

  if (!query.data) {
    return <PreviewBox>{t('nodes.generatorLoading')}</PreviewBox>;
  }

  if (query.data.error) {
    return <PreviewBox tone="fg.error">{query.data.error}</PreviewBox>;
  }

  const items = query.data.items;

  if (items.length === 0) {
    return <PreviewBox>{`<${t('nodes.generatorNoValues')}>`}</PreviewBox>;
  }

  return (
    <PreviewBox>
      {request.kind === 'boardImages'
        ? `<${t('nodes.generatorImages', { count: items.length })}>`
        : items.slice(0, PREVIEW_ITEM_LIMIT).map(formatItem).join(', ') +
          (items.length > PREVIEW_ITEM_LIMIT
            ? ` … ${t('nodes.generatorPreviewTruncated', { count: items.length })}`
            : '')}
    </PreviewBox>
  );
};

/**
 * What the generator will produce. Synchronous variants derive in render; async ones query, on the value as it
 * stood before the current burst of typing until the burst settles.
 */
const GeneratorPreview = ({
  generator,
  settled,
}: {
  generator: WorkflowGeneratorValue;
  settled: WorkflowGeneratorValue;
}) => {
  const { t } = useTranslation();
  const invalidReason = getWorkflowGeneratorInvalidReason(generator);

  if (invalidReason !== null) {
    return <PreviewBox tone="fg.error">{invalidReason}</PreviewBox>;
  }

  const requested = getWorkflowGeneratorRequestedCount(settled);

  // Never build an oversized list for a preview; readiness refuses it on the same rule.
  if (requested !== null && requested > WORKFLOW_BATCH_MAX_ITEMS) {
    return (
      <PreviewBox tone="fg.error">
        {t('nodes.generatorTooMany', { max: WORKFLOW_BATCH_MAX_ITEMS.toLocaleString() })}
      </PreviewBox>
    );
  }

  if (isRandomWorkflowGeneratorValue(generator)) {
    const count = 'count' in generator && generator.count !== null ? generator.count : 0;

    return <PreviewBox>{`<${t('nodes.generatorNRandomValues', { count })}>`}</PreviewBox>;
  }

  const resolution = resolveWorkflowGeneratorValue(settled);

  if (resolution.kind === 'async') {
    return <AsyncPreview request={resolution.request} />;
  }

  if (resolution.items.length === 0) {
    return <PreviewBox>{`<${t('nodes.generatorNoValues')}>`}</PreviewBox>;
  }

  return (
    <PreviewBox>
      {resolution.items.slice(0, PREVIEW_ITEM_LIMIT).map(formatItem).join(', ') +
        (resolution.items.length > PREVIEW_ITEM_LIMIT
          ? ` … ${t('nodes.generatorPreviewTruncated', { count: resolution.items.length })}`
          : '')}
    </PreviewBox>
  );
};

/** The generator widget: a variant picker, the variant's settings, and a preview of what it yields. */
export const GeneratorFieldInput = ({ id, invalid, onChange, template, value }: WorkflowFieldInputProps) => {
  const { t } = useTranslation();
  const typeName = template.type.name;
  const parsed = parseWorkflowGeneratorValue(typeName, value);
  // An unreadable stored value still shows its own variant's defaults, so the user's variant choice survives.
  const storedVariant = typeof value === 'object' && value !== null ? (value as { type?: unknown }).type : undefined;
  const generator =
    parsed ??
    (typeof storedVariant === 'string' &&
    isWorkflowGeneratorVariant(storedVariant) &&
    (WORKFLOW_GENERATOR_VARIANTS as Record<string, readonly string[]>)[typeName]?.includes(storedVariant)
      ? getWorkflowGeneratorVariantDefaults(storedVariant)
      : getDefaultWorkflowGeneratorValue(typeName));
  const variantCollection = useMemo(() => {
    const variants =
      (WORKFLOW_GENERATOR_VARIANTS as Record<string, readonly WorkflowGeneratorVariant[]>)[typeName] ?? [];

    return createListCollection({
      items: variants.map((variant) => ({ label: t(VARIANT_LABEL_KEYS[variant]), value: variant })),
    });
  }, [t, typeName]);
  const categoryCollection = useMemo(
    () =>
      createListCollection({
        items: CATEGORY_OPTIONS.map((option) => ({ label: t(option.labelKey), value: option.value })),
      }),
    [t]
  );
  const prefix = id ?? 'generator';

  // A typing burst previews the value it started from until it settles; undo and picks land immediately.
  const [held, setHeld] = useState<WorkflowGeneratorValue | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useMountEffect(() => () => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
    }
  });

  const commit = useCallback(
    (next: WorkflowGeneratorValue, { debounce = false } = {}) => {
      if (debounce && generator) {
        if (timerRef.current === null) {
          setHeld(generator);
        } else {
          clearTimeout(timerRef.current);
        }

        timerRef.current = setTimeout(() => {
          timerRef.current = null;
          setHeld(null);
        }, PREVIEW_DEBOUNCE_MS);
      }

      // A legacy resolved override would hide the edit; drop it whenever the settings change.
      const { values: _values, ...rest } = next as WorkflowGeneratorValue & { values?: number[] };

      onChange(rest);
    },
    [generator, onChange]
  );
  const setProperty = useCallback(
    (key: string, propertyValue: unknown, debounce = false) => {
      if (generator) {
        commit({ ...generator, [key]: propertyValue } as WorkflowGeneratorValue, { debounce });
      }
    },
    [commit, generator]
  );
  const onVariantChange = useCallback(
    ({ value: next }: { value: string[] }) => {
      const variant = next[0] as WorkflowGeneratorVariant | undefined;

      if (variant) {
        commit(getWorkflowGeneratorVariantDefaults(variant));
      }
    },
    [commit]
  );
  const onCategoryChange = useCallback(
    ({ value: next }: { value: string[] }) => setProperty('category', next[0]),
    [setProperty]
  );
  const onBoardChange = useCallback((boardId: string | undefined) => setProperty('board_id', boardId), [setProperty]);
  const onInputChange = useCallback((text: string) => setProperty('input', text, true), [setProperty]);
  const onSplitOnChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setProperty('splitOn', event.currentTarget.value, true),
    [setProperty]
  );
  const onSeedChange = useCallback((seed: number | null) => setProperty('seed', seed), [setProperty]);
  const number = (key: string) => (next: number | null) => setProperty(key, next, true);
  const isOff = (key: string) =>
    generator !== undefined && getWorkflowGeneratorPropertyInvalidReason(generator, key) !== null;
  const variantType = generator?.type ?? '';
  const variantValue = useMemo(() => [variantType], [variantType]);
  const category = generator?.type === 'image_generator_images_from_board' ? generator.category : 'images';
  const categoryValue = useMemo(() => [category], [category]);

  if (!generator) {
    return null;
  }

  return (
    <Stack gap="1.5" w="full">
      <Select
        aria-label={template.title}
        className="nodrag"
        collection={variantCollection}
        invalid={invalid}
        size="xs"
        value={variantValue}
        valueTextProps={SELECT_VALUE_TEXT_PROPS}
        w="full"
        onValueChange={onVariantChange}
      />
      {generator.type === 'float_generator_arithmetic_sequence' ||
      generator.type === 'integer_generator_arithmetic_sequence' ? (
        <HStack gap="2">
          <NumberSetting
            id={`${prefix}-start`}
            invalid={isOff('start')}
            integer={generator.type.startsWith('integer')}
            label={t('nodes.generatorStart')}
            value={generator.start}
            onCommit={number('start')}
          />
          <NumberSetting
            id={`${prefix}-step`}
            invalid={isOff('step')}
            integer={generator.type.startsWith('integer')}
            label={t('nodes.generatorStep')}
            value={generator.step}
            onCommit={number('step')}
          />
          <NumberSetting
            id={`${prefix}-count`}
            invalid={isOff('count')}
            integer
            label={t('nodes.generatorCount')}
            min={1}
            value={generator.count}
            onCommit={number('count')}
          />
        </HStack>
      ) : null}
      {generator.type === 'float_generator_linear_distribution' ||
      generator.type === 'integer_generator_linear_distribution' ? (
        <HStack gap="2">
          <NumberSetting
            id={`${prefix}-start`}
            invalid={isOff('start')}
            integer={generator.type.startsWith('integer')}
            label={t('nodes.generatorStart')}
            value={generator.start}
            onCommit={number('start')}
          />
          <NumberSetting
            id={`${prefix}-end`}
            invalid={isOff('end')}
            integer={generator.type.startsWith('integer')}
            label={t('nodes.generatorEnd')}
            value={generator.end}
            onCommit={number('end')}
          />
          <NumberSetting
            id={`${prefix}-count`}
            invalid={isOff('count')}
            integer
            label={t('nodes.generatorCount')}
            min={1}
            value={generator.count}
            onCommit={number('count')}
          />
        </HStack>
      ) : null}
      {generator.type === 'float_generator_random_distribution_uniform' ||
      generator.type === 'integer_generator_random_distribution_uniform' ? (
        <Stack gap="1.5">
          <HStack gap="2">
            <NumberSetting
              id={`${prefix}-min`}
              invalid={isOff('min')}
              integer={generator.type.startsWith('integer')}
              label={t('nodes.generatorMin')}
              value={generator.min}
              onCommit={number('min')}
            />
            <NumberSetting
              id={`${prefix}-max`}
              invalid={isOff('max')}
              integer={generator.type.startsWith('integer')}
              label={t('nodes.generatorMax')}
              value={generator.max}
              onCommit={number('max')}
            />
            <NumberSetting
              id={`${prefix}-count`}
              invalid={isOff('count')}
              integer
              label={t('nodes.generatorCount')}
              min={1}
              value={generator.count}
              onCommit={number('count')}
            />
          </HStack>
          <SeedSetting id={prefix} invalid={isOff('seed')} seed={generator.seed} onCommit={onSeedChange} />
        </Stack>
      ) : null}
      {generator.type === 'float_generator_parse_string' ||
      generator.type === 'integer_generator_parse_string' ||
      generator.type === 'string_generator_parse_string' ? (
        <Stack gap="1.5">
          <HStack gap="1">
            <chakra.label color="fg.subtle" flexShrink={0} fontSize="2xs" htmlFor={`${prefix}-split-on`}>
              {t('nodes.splitOn')}
            </chakra.label>
            <Input
              className="nodrag"
              id={`${prefix}-split-on`}
              size="2xs"
              value={generator.splitOn}
              onChange={onSplitOnChange}
            />
          </HStack>
          <InputSetting id={`${prefix}-input`} value={generator.input} onCommit={onInputChange} />
        </Stack>
      ) : null}
      {generator.type === 'string_generator_dynamic_prompts_random' ? (
        <Stack gap="1.5">
          <HStack gap="2">
            <NumberSetting
              id={`${prefix}-count`}
              invalid={isOff('count')}
              integer
              label={t('nodes.generatorCount')}
              max={WORKFLOW_DYNAMIC_PROMPTS_MAX}
              min={1}
              value={generator.count}
              onCommit={number('count')}
            />
            <SeedSetting id={prefix} invalid={isOff('seed')} seed={generator.seed} onCommit={onSeedChange} />
          </HStack>
          <InputSetting id={`${prefix}-input`} value={generator.input} onCommit={onInputChange} />
        </Stack>
      ) : null}
      {generator.type === 'string_generator_dynamic_prompts_combinatorial' ? (
        <Stack gap="1.5">
          <NumberSetting
            id={`${prefix}-max-prompts`}
            invalid={isOff('maxPrompts')}
            integer
            label={t('nodes.generatorMaxPrompts')}
            max={WORKFLOW_DYNAMIC_PROMPTS_MAX}
            min={1}
            value={generator.maxPrompts}
            onCommit={number('maxPrompts')}
          />
          <InputSetting id={`${prefix}-input`} value={generator.input} onCommit={onInputChange} />
        </Stack>
      ) : null}
      {generator.type === 'image_generator_images_from_board' ? (
        <Stack gap="1.5">
          <BoardSetting boardId={generator.board_id} id={`${prefix}-board`} onCommit={onBoardChange} />
          <Field.Root gap="0" minW="0" w="full">
            <Select
              aria-label={t('nodes.generatorImagesCategory')}
              className="nodrag"
              collection={categoryCollection}
              size="xs"
              value={categoryValue}
              valueTextProps={SELECT_VALUE_TEXT_PROPS}
              w="full"
              onValueChange={onCategoryChange}
            />
          </Field.Root>
        </Stack>
      ) : null}
      <GeneratorPreview generator={generator} settled={held ?? generator} />
    </Stack>
  );
};
