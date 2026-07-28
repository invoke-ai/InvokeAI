import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';
import type { DynamicPromptsExpansion } from '@features/generation/ui/useDynamicPrompts';

import { HStack, NumberInput, SegmentGroup, Stack, Text } from '@chakra-ui/react';
import {
  createDynamicPromptsSampleSeed,
  DYNAMIC_PROMPTS_MAX_PROMPTS,
  DYNAMIC_PROMPTS_MIN_PROMPTS,
  sanitizeMaxPrompts,
} from '@features/generation/core/dynamicPrompts';
import { HighlightedPrompt } from '@features/generation/ui/promptFields/PromptHighlight';
import { Button, IconButton } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { ShuffleIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/** Rendering every row of a 10,000-prompt expansion would cost more than it tells the user. */
const MAX_PREVIEW_ROWS = 200;
const TABULAR_NUMS = { fontVariantNumeric: 'tabular-nums' } as const;

export interface DynamicPromptsFieldConfig extends DynamicPromptsConfig {
  onChange: (patch: Partial<DynamicPromptsConfig>) => void;
}

export const DynamicPromptsPanel = ({
  batchCount,
  config,
  expansion,
  onUsePrompt,
  showSyntaxHighlighting,
}: {
  batchCount: number;
  config: DynamicPromptsFieldConfig;
  expansion: DynamicPromptsExpansion;
  showSyntaxHighlighting: boolean;
  onUsePrompt: (prompt: string) => void;
}) => {
  const { t } = useTranslation();
  const { onChange } = config;
  const visiblePrompts = expansion.prompts.slice(0, MAX_PREVIEW_ROWS);
  const hiddenPromptCount = expansion.prompts.length - visiblePrompts.length;

  const handleModeChange = useCallback(
    (event: { value: string | null }) => onChange({ combinatorial: event.value !== 'random' }),
    [onChange]
  );
  const handleMaxPromptsChange = useCallback(
    ({ valueAsNumber }: { valueAsNumber: number }) => {
      if (Number.isFinite(valueAsNumber)) {
        onChange({ maxPrompts: sanitizeMaxPrompts(valueAsNumber) });
      }
    },
    [onChange]
  );
  const handleSeedBehaviourChange = useCallback(
    (event: { value: string | null }) =>
      onChange({ seedBehaviour: event.value === 'per-image' ? 'per-image' : 'per-iteration' }),
    [onChange]
  );
  const handleShuffle = useCallback(() => onChange({ sampleSeed: createDynamicPromptsSampleSeed() }), [onChange]);

  const modeItems = useMemo(
    () => [
      { label: t('widgets.generate.dynamicPrompts.allCombinations'), value: 'all' },
      { label: t('widgets.generate.dynamicPrompts.randomSample'), value: 'random' },
    ],
    [t]
  );
  const seedItems = useMemo(
    () => [
      { label: t('widgets.generate.dynamicPrompts.seedPerIteration'), value: 'per-iteration' },
      { label: t('widgets.generate.dynamicPrompts.seedPerImage'), value: 'per-image' },
    ],
    [t]
  );
  const maxPromptsLabel = config.combinatorial
    ? t('widgets.generate.dynamicPrompts.maxPrompts')
    : t('widgets.generate.dynamicPrompts.numberOfPrompts');

  return (
    <Stack gap="2.5">
      <HStack justify="space-between">
        <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
          {t('widgets.generate.dynamicPrompts.title')}
        </Text>
        <Text color="fg.muted" css={TABULAR_NUMS} fontSize="2xs">
          {expansion.isLoading
            ? t('widgets.generate.dynamicPrompts.expanding')
            : t('widgets.generate.dynamicPrompts.summary', {
                generations: expansion.count * batchCount,
                iterations: batchCount,
                prompts: expansion.count,
              })}
        </Text>
      </HStack>

      {/* Shuffle only means anything for a random sample, but it is always
          rendered — merely hidden — so switching modes cannot change the row's
          shape. `visibility: hidden` also takes it out of the tab order. */}
      <HStack gap="2">
        <SegmentGroup.Root
          flex="1"
          size="xs"
          value={config.combinatorial ? 'all' : 'random'}
          onValueChange={handleModeChange}
        >
          <SegmentGroup.Indicator />
          <SegmentGroup.Items items={modeItems} />
        </SegmentGroup.Root>
        <Tooltip content={t('widgets.generate.dynamicPrompts.shuffle')}>
          <IconButton
            aria-label={t('widgets.generate.dynamicPrompts.shuffle')}
            size="2xs"
            variant="ghost"
            visibility={config.combinatorial ? 'hidden' : 'visible'}
            onClick={handleShuffle}
          >
            <ShuffleIcon />
          </IconButton>
        </Tooltip>
      </HStack>

      <HStack align="end" gap="2">
        <Stack flex="1" gap="1" minW="0">
          <Text color="fg.subtle" fontSize="2xs">
            {maxPromptsLabel}
          </Text>
          <NumberInput.Root
            allowMouseWheel
            max={DYNAMIC_PROMPTS_MAX_PROMPTS}
            min={DYNAMIC_PROMPTS_MIN_PROMPTS}
            size="xs"
            value={String(config.maxPrompts)}
            onValueChange={handleMaxPromptsChange}
          >
            <NumberInput.Control />
            <NumberInput.Input aria-label={maxPromptsLabel} paddingStart="2" />
          </NumberInput.Root>
        </Stack>
        <Stack flex="1" gap="1" minW="0">
          <Text color="fg.subtle" fontSize="2xs">
            {t('widgets.generate.dynamicPrompts.seedBehaviour')}
          </Text>
          {/* A two-option choice reads better as a segmented control than a dropdown:
              one click instead of two, and it matches the mode control above. */}
          <SegmentGroup.Root
            aria-label={t('widgets.generate.dynamicPrompts.seedBehaviour')}
            size="xs"
            value={config.seedBehaviour}
            onValueChange={handleSeedBehaviourChange}
          >
            <SegmentGroup.Indicator />
            <SegmentGroup.Items items={seedItems} />
          </SegmentGroup.Root>
        </Stack>
      </HStack>

      {expansion.isError ? (
        <Text color="fg.error" fontSize="2xs">
          {t('widgets.generate.dynamicPrompts.problemGeneratingPrompts')}
        </Text>
      ) : expansion.error ? (
        <Text color="fg.error" fontSize="2xs" wordBreak="break-word">
          {expansion.error}
        </Text>
      ) : null}

      <Scrollable h="14rem" label={t('widgets.generate.dynamicPrompts.promptsPreview')}>
        <Stack gap="0">
          {visiblePrompts.map((prompt, index) => (
            <DynamicPromptRow
              key={`${index}-${prompt}`}
              index={index}
              prompt={prompt}
              showSyntaxHighlighting={showSyntaxHighlighting}
              onUsePrompt={onUsePrompt}
            />
          ))}
          {hiddenPromptCount > 0 ? (
            <Text color="fg.subtle" fontSize="2xs" px="2" py="1.5">
              {t('widgets.generate.dynamicPrompts.andMore', { count: hiddenPromptCount })}
            </Text>
          ) : null}
        </Stack>
      </Scrollable>
    </Stack>
  );
};

const DynamicPromptRow = ({
  index,
  onUsePrompt,
  prompt,
  showSyntaxHighlighting,
}: {
  index: number;
  prompt: string;
  showSyntaxHighlighting: boolean;
  onUsePrompt: (prompt: string) => void;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onUsePrompt(prompt), [onUsePrompt, prompt]);

  return (
    <Button
      alignItems="start"
      gap="2"
      h="auto"
      justifyContent="start"
      px="2"
      py="1.5"
      size="xs"
      title={t('widgets.generate.dynamicPrompts.usePrompt')}
      transitionDuration="faster"
      variant="ghost"
      onClick={handleClick}
    >
      <Text as="span" color="fg.subtle" css={TABULAR_NUMS} fontSize="2xs">
        {index + 1}
      </Text>
      <Text as="span" color="fg" fontFamily="mono" fontSize="0.72rem" textAlign="start" wordBreak="break-word">
        {/* An expanded prompt has no dynamic syntax left in it, so the useful
            colouring here is attention and embeddings — the defaults. */}
        <HighlightedPrompt enabled={showSyntaxHighlighting} prompt={prompt} />
      </Text>
    </Button>
  );
};
