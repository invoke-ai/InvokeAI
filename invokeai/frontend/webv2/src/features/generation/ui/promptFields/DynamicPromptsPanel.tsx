import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';
import type { DynamicPromptsExpansion } from '@features/generation/ui/useDynamicPrompts';

import { Badge, HStack, Menu, NumberInput, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import {
  createDynamicPromptsSampleSeed,
  DYNAMIC_PROMPTS_MAX_PROMPTS,
  DYNAMIC_PROMPTS_MIN_PROMPTS,
  sanitizeMaxPrompts,
} from '@features/generation/core/dynamicPrompts';
import { HighlightedPrompt } from '@features/generation/ui/promptFields/PromptHighlight';
import { PANEL_HEADER_CONTROL_HEIGHT, PromptPanelHeader } from '@features/generation/ui/promptFields/PromptPanelHeader';
import { Button, IconButton } from '@platform/ui/Button';
import { Field } from '@platform/ui/Field';
import { MenuContent } from '@platform/ui/Menu';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { ChevronDownIcon, ShuffleIcon } from 'lucide-react';
import { useCallback, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/** Rendering every row of a 10,000-prompt expansion would cost more than it tells the user. */
const MAX_PREVIEW_ROWS = 200;
const TABULAR_NUMS = { fontVariantNumeric: 'tabular-nums' } as const;
const MENU_POSITIONING = { placement: 'bottom-start' } as const;
const SWITCH_CHECKED = { bg: 'accent.solid' } as const;

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
  // Chakra derives the switch's label/input ids from its own counter, which can
  // collide with the number input beside it and send label clicks to the wrong
  // control. Explicit ids keep them apart.
  const seedSwitchId = useId();
  const modeFieldId = useId();
  const modeTriggerId = useId();
  // A <label for> cannot name a button, so the trigger is named by the field's
  // label plus its own text: "Mode" + "All combinations".
  const modeLabelledBy = `${modeFieldId}-label ${modeTriggerId}`;
  const modeMenuIds = useMemo(() => ({ trigger: modeTriggerId }), [modeTriggerId]);
  const seedSwitchIds = useMemo(() => ({ hiddenInput: seedSwitchId, label: `${seedSwitchId}-label` }), [seedSwitchId]);
  const visiblePrompts = expansion.prompts.slice(0, MAX_PREVIEW_ROWS);
  const hiddenPromptCount = expansion.prompts.length - visiblePrompts.length;

  const handleModeChange = useCallback(
    (event: { value: string }) => onChange({ combinatorial: event.value !== 'random' }),
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
    (event: { checked: boolean }) => onChange({ seedBehaviour: event.checked ? 'per-image' : 'per-iteration' }),
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
  const mode = config.combinatorial ? 'all' : 'random';
  const modeValue = useMemo(() => [mode], [mode]);
  const modeLabel = modeItems.find((item) => item.value === mode)?.label ?? '';
  const maxPromptsLabel = config.combinatorial
    ? t('widgets.generate.dynamicPrompts.maxPrompts')
    : t('widgets.generate.dynamicPrompts.numberOfPrompts');

  return (
    <Stack gap="2.5">
      <PromptPanelHeader label={t('widgets.generate.dynamicPrompts.title')}>
        {/* Monospace rather than inline icons: lucide's X is a close glyph, not a
            times sign, and there is no arithmetic multiply in the set. Mono with
            tabular figures also stops the badge jittering as the counts change. */}
        <Badge
          color="fg.muted"
          css={TABULAR_NUMS}
          fontFamily="mono"
          fontSize="2xs"
          fontWeight="500"
          h={PANEL_HEADER_CONTROL_HEIGHT}
          px="1.5"
          variant="subtle"
        >
          {expansion.isLoading
            ? t('widgets.generate.dynamicPrompts.expanding')
            : t('widgets.generate.dynamicPrompts.summary', {
                generations: expansion.count * batchCount,
                iterations: batchCount,
                prompts: expansion.count,
              })}
        </Badge>
      </PromptPanelHeader>

      <HStack align="end" gap="2">
        <Field id={modeFieldId} label={t('widgets.generate.dynamicPrompts.mode')}>
          {/* A menu rather than a Select: Chakra's Select renders a hidden native
              select whose sync throws inside this popover. */}
          <Menu.Root ids={modeMenuIds} positioning={MENU_POSITIONING}>
            <Menu.Trigger asChild>
              <Button
                aria-labelledby={modeLabelledBy}
                justifyContent="space-between"
                minW="0"
                size="xs"
                variant="outline"
                w="full"
              >
                <Text as="span" truncate>
                  {modeLabel}
                </Text>
                <ChevronDownIcon />
              </Button>
            </Menu.Trigger>
            <Portal>
              <Menu.Positioner>
                <MenuContent minW="10rem" py="1">
                  <Menu.RadioItemGroup value={modeValue[0]} onValueChange={handleModeChange}>
                    {modeItems.map((item) => (
                      <Menu.RadioItem key={item.value} value={item.value}>
                        <Menu.ItemText>{item.label}</Menu.ItemText>
                        <Menu.ItemIndicator />
                      </Menu.RadioItem>
                    ))}
                  </Menu.RadioItemGroup>
                </MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        </Field>
        {/* Fixed width and no shrink: the mode field takes the slack instead. */}
        <Field flex="0 0 auto" label={maxPromptsLabel} w="6.5rem">
          <NumberInput.Root
            allowMouseWheel
            max={DYNAMIC_PROMPTS_MAX_PROMPTS}
            min={DYNAMIC_PROMPTS_MIN_PROMPTS}
            size="xs"
            value={String(config.maxPrompts)}
            onValueChange={handleMaxPromptsChange}
          >
            <NumberInput.Control />
            <NumberInput.Input paddingStart="2" />
          </NumberInput.Root>
        </Field>
        {/* Always rendered, merely hidden, so switching modes cannot reflow the row. */}
        <Tooltip content={t('widgets.generate.dynamicPrompts.shuffle')}>
          <IconButton
            aria-label={t('widgets.generate.dynamicPrompts.shuffle')}
            size="xs"
            variant="ghost"
            visibility={config.combinatorial ? 'hidden' : 'visible'}
            onClick={handleShuffle}
          >
            <ShuffleIcon />
          </IconButton>
        </Tooltip>
      </HStack>

      <Switch.Root
        checked={config.seedBehaviour === 'per-image'}
        ids={seedSwitchIds}
        size="sm"
        onCheckedChange={handleSeedBehaviourChange}
      >
        <Switch.HiddenInput />
        <Switch.Control _checked={SWITCH_CHECKED}>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label color="fg.muted" fontSize="2xs">
          {t('widgets.generate.dynamicPrompts.newSeedPerImage')}
        </Switch.Label>
      </Switch.Root>

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
