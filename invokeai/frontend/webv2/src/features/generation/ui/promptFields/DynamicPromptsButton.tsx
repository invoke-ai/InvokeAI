/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { DynamicPromptsConfig, DynamicPromptsSeedBehaviour } from '@features/generation/core/dynamicPrompts';

import {
  createListCollection,
  HStack,
  NumberInput,
  Popover,
  Portal,
  SegmentGroup,
  Stack,
  Text,
} from '@chakra-ui/react';
import {
  createDynamicPromptsSampleSeed,
  DYNAMIC_PROMPTS_MAX_PROMPTS,
  DYNAMIC_PROMPTS_MIN_PROMPTS,
  isDynamicPromptsSeedBehaviour,
  sanitizeMaxPrompts,
} from '@features/generation/core/dynamicPrompts';
import { WildcardsPanel } from '@features/generation/ui/promptFields/WildcardsPanel';
import { useDynamicPrompts } from '@features/generation/ui/useDynamicPrompts';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { Button, IconButton } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { Select } from '@platform/ui/Select';
import { Tooltip } from '@platform/ui/Tooltip';
import { BracesIcon, ShuffleIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Rendering every row of a 10,000-prompt expansion would cost more than it tells the user. */
const MAX_PREVIEW_ROWS = 200;
const POPOVER_POSITIONING_BOTTOM_END = { placement: 'bottom-end' } as const;
const COUNT_TEXT_PROPS = { fontVariantNumeric: 'tabular-nums' } as const;

export interface DynamicPromptsFieldConfig extends DynamicPromptsConfig {
  onChange: (patch: Partial<DynamicPromptsConfig>) => void;
}

interface DynamicPromptsButtonProps {
  config: DynamicPromptsFieldConfig;
  batchCount: number;
  positivePrompt: string;
  onUsePrompt: (prompt: string) => void;
  onInsertText: (text: string) => void;
}

export const DynamicPromptsButton = ({
  batchCount,
  config,
  onInsertText,
  onUsePrompt,
  positivePrompt,
}: DynamicPromptsButtonProps) => {
  const { t } = useTranslation();
  const triggerId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [tab, setTab] = useState<'preview' | 'wildcards'>('preview');
  const expansion = useDynamicPrompts(positivePrompt, config);
  const catalog = useWildcards();
  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  const handleOpenChange = useCallback((event: { open: boolean }) => setIsOpen(event.open), []);
  const handleUsePrompt = useCallback(
    (prompt: string) => {
      onUsePrompt(prompt);
      setIsOpen(false);
    },
    [onUsePrompt]
  );
  const handleInsert = useCallback(
    (text: string) => {
      onInsertText(text);
      setIsOpen(false);
    },
    [onInsertText]
  );
  const handleTabChange = useCallback(
    (event: { value: string | null }) => setTab(event.value === 'wildcards' ? 'wildcards' : 'preview'),
    []
  );
  const tabItems = useMemo(
    () => [
      { label: t('widgets.generate.dynamicPrompts.preview'), value: 'preview' },
      { label: t('widgets.generate.dynamicPrompts.wildcards'), value: 'wildcards' },
    ],
    [t]
  );

  const tooltip = expansion.isDynamic
    ? t('widgets.generate.dynamicPrompts.showPrompts')
    : t('widgets.generate.dynamicPrompts.noDynamicSyntax');
  // Quiet states only: an em-dash while the expansion is in flight, an error
  // tint when it failed. No spinner, no animation on a control this small.
  const countLabel = !expansion.isDynamic ? null : expansion.isLoading ? '—' : String(expansion.count);

  return (
    <Popover.Root
      ids={popoverIds}
      lazyMount
      open={isOpen}
      positioning={POPOVER_POSITIONING_BOTTOM_END}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Tooltip content={tooltip} ids={popoverIds}>
        <Popover.Trigger asChild>
          <IconButton
            aria-label={t('widgets.generate.dynamicPrompts.showPrompts')}
            color={expansion.isError ? 'fg.error' : undefined}
            opacity={expansion.isDynamic ? undefined : 0.5}
            px={countLabel ? '1' : undefined}
            size="2xs"
            variant="ghost"
            w={countLabel ? 'auto' : undefined}
          >
            <BracesIcon />
            {countLabel ? (
              <Text as="span" css={COUNT_TEXT_PROPS} fontSize="2xs">
                {countLabel}
              </Text>
            ) : null}
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="26rem">
            <Popover.Body p="2.5">
              <Stack gap="2.5">
                <SegmentGroup.Root size="xs" value={tab} onValueChange={handleTabChange}>
                  <SegmentGroup.Indicator />
                  <SegmentGroup.Items items={tabItems} />
                </SegmentGroup.Root>
                {tab === 'preview' ? (
                  <DynamicPromptsPanel
                    batchCount={batchCount}
                    config={config}
                    expansion={expansion}
                    onUsePrompt={handleUsePrompt}
                  />
                ) : (
                  <WildcardsPanel catalog={catalog} onInsert={handleInsert} />
                )}
              </Stack>
            </Popover.Body>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

const DynamicPromptsPanel = ({
  batchCount,
  config,
  expansion,
  onUsePrompt,
}: {
  batchCount: number;
  config: DynamicPromptsFieldConfig;
  expansion: ReturnType<typeof useDynamicPrompts>;
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
    ({ valueAsNumber }: { valueAsNumber: number }) =>
      Number.isFinite(valueAsNumber) && onChange({ maxPrompts: sanitizeMaxPrompts(valueAsNumber) }),
    [onChange]
  );
  const handleSeedBehaviourChange = useCallback(
    ({ value }: { value: string[] }) =>
      isDynamicPromptsSeedBehaviour(value[0]) && onChange({ seedBehaviour: value[0] as DynamicPromptsSeedBehaviour }),
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
  const maxPromptsLabel = config.combinatorial
    ? t('widgets.generate.dynamicPrompts.maxPrompts')
    : t('widgets.generate.dynamicPrompts.numberOfPrompts');
  const seedBehaviourValue = useMemo(() => [config.seedBehaviour], [config.seedBehaviour]);
  const seedBehaviourCollection = useMemo(
    () =>
      createListCollection({
        items: [
          { label: t('widgets.generate.dynamicPrompts.seedPerIteration'), value: 'per-iteration' },
          { label: t('widgets.generate.dynamicPrompts.seedPerImage'), value: 'per-image' },
        ],
      }),
    [t]
  );

  return (
    <Stack gap="2.5">
      <HStack justify="space-between">
        <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
          {t('widgets.generate.dynamicPrompts.title')}
        </Text>
        <Text color="fg.muted" css={COUNT_TEXT_PROPS} fontSize="2xs">
          {expansion.isLoading
            ? t('widgets.generate.dynamicPrompts.expanding')
            : t('widgets.generate.dynamicPrompts.summary', {
                generations: expansion.count * batchCount,
                iterations: batchCount,
                prompts: expansion.count,
              })}
        </Text>
      </HStack>

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
        {config.combinatorial ? null : (
          <Tooltip content={t('widgets.generate.dynamicPrompts.shuffle')}>
            <IconButton
              aria-label={t('widgets.generate.dynamicPrompts.shuffle')}
              size="xs"
              variant="ghost"
              onClick={handleShuffle}
            >
              <ShuffleIcon />
            </IconButton>
          </Tooltip>
        )}
      </HStack>

      <HStack gap="2">
        <Stack flex="1" gap="1">
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
        <Stack flex="1" gap="1">
          <Text color="fg.subtle" fontSize="2xs">
            {t('widgets.generate.dynamicPrompts.seedBehaviour')}
          </Text>
          <Select
            aria-label={t('widgets.generate.dynamicPrompts.seedBehaviour')}
            collection={seedBehaviourCollection}
            size="xs"
            value={seedBehaviourValue}
            onValueChange={handleSeedBehaviourChange}
          />
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
            <DynamicPromptRow key={`${index}-${prompt}`} index={index} prompt={prompt} onUsePrompt={onUsePrompt} />
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
}: {
  index: number;
  prompt: string;
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
      <Text as="span" color="fg.subtle" css={COUNT_TEXT_PROPS} fontSize="2xs">
        {index + 1}
      </Text>
      <Text as="span" color="fg" fontFamily="mono" fontSize="0.72rem" textAlign="start" wordBreak="break-word">
        {prompt}
      </Text>
    </Button>
  );
};
