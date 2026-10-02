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
import { Row } from '@platform/ui/Row';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { ChevronDownIcon, ShuffleIcon } from 'lucide-react';
import { useCallback, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/** Rendering every row of a 10,000-prompt expansion would cost more than it tells the user. */
const MAX_PREVIEW_ROWS = 200;
const TABULAR_NUMS = { fontVariantNumeric: 'tabular-nums' } as const;
// Use a hover fill distinct from the popover's muted surface.
const PROMPT_ROW_HOVER_PROPS = { bg: 'bg.emphasized/60' } as const;
// Keep preview text readable while interactions are disabled.
const DISABLED_PROMPT_ROW_PROPS = { cursor: 'default', opacity: 1 } as const;
const NO_HOVER_PROPS = { bg: 'transparent' } as const;
const MENU_POSITIONING = { placement: 'bottom-start' } as const;
const SWITCH_CHECKED = { bg: 'accent.solid' } as const;

export interface DynamicPromptsFieldConfig extends DynamicPromptsConfig {
  /** A fixed seed applies to every image, so per-image sharing has nothing left to decide. */
  isSeedFixed: boolean;
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
  // Explicit sibling input IDs prevent labels from targeting the wrong control.
  const seedSwitchId = useId();
  const modeFieldId = useId();
  const modeTriggerId = useId();
  // aria-labelledby combines the field label and value for the composite trigger's accessible name.
  const modeLabelledBy = `${modeFieldId}-label ${modeTriggerId}`;
  const modeMenuIds = useMemo(() => ({ trigger: modeTriggerId }), [modeTriggerId]);
  const seedSwitchIds = useMemo(() => ({ hiddenInput: seedSwitchId, label: `${seedSwitchId}-label` }), [seedSwitchId]);
  const seedHeldNoteId = `${seedSwitchId}-held`;
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
  const modeLabel = modeItems.find((item) => item.value === mode)?.label ?? '';
  const maxPromptsLabel = config.combinatorial
    ? t('widgets.generate.dynamicPrompts.maxPrompts')
    : t('widgets.generate.dynamicPrompts.numberOfPrompts');

  return (
    <Stack gap="2.5">
      <PromptPanelHeader label={t('widgets.generate.dynamicPrompts.title')}>
        {/* Use a literal multiplication sign and tabular figures. */}
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
          {/* Use a menu to avoid Select's hidden-native synchronization failure. */}
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
                  <Menu.RadioItemGroup value={mode} onValueChange={handleModeChange}>
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
        disabled={config.isSeedFixed}
        ids={seedSwitchIds}
        size="sm"
        onCheckedChange={handleSeedBehaviourChange}
      >
        <Switch.HiddenInput aria-describedby={config.isSeedFixed ? seedHeldNoteId : undefined} />
        <Switch.Control _checked={SWITCH_CHECKED}>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label color="fg.muted" fontSize="2xs">
          {t('widgets.generate.dynamicPrompts.newSeedPerImage')}
        </Switch.Label>
      </Switch.Root>
      {config.isSeedFixed ? (
        <Text color="fg.subtle" fontSize="2xs" id={seedHeldNoteId} mt="-1">
          {t('widgets.generate.dynamicPrompts.seedHeldForEveryImage')}
        </Text>
      ) : null}

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
              // A single unchanged expansion is read-only because applying it is a no-op.
              isDisabled={expansion.prompts.length === 1}
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
  isDisabled,
  onUsePrompt,
  prompt,
  showSyntaxHighlighting,
}: {
  index: number;
  isDisabled: boolean;
  prompt: string;
  showSyntaxHighlighting: boolean;
  onUsePrompt: (prompt: string) => void;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onUsePrompt(prompt), [onUsePrompt, prompt]);

  return (
    <Row
      alignItems="start"
      asChild
      borderColor="transparent"
      borderWidth="1px"
      fontWeight="medium"
      gap="2"
      h="auto"
      justifyContent="start"
      px="2"
      py="1.5"
      textStyle="xs"
      title={isDisabled ? undefined : t('widgets.generate.dynamicPrompts.usePrompt')}
      whiteSpace="nowrap"
      _disabled={DISABLED_PROMPT_ROW_PROPS}
      _hover={isDisabled ? NO_HOVER_PROPS : PROMPT_ROW_HOVER_PROPS}
    >
      <button disabled={isDisabled} type="button" onClick={handleClick}>
        <Text as="span" color="fg.subtle" css={TABULAR_NUMS} fontSize="2xs">
          {index + 1}
        </Text>
        <Text as="span" color="fg" fontFamily="mono" fontSize="0.72rem" textAlign="start" wordBreak="break-word">
          {/* Expanded prompts highlight attention/embeddings only; dynamic syntax has already been consumed. */}
          <HighlightedPrompt enabled={showSyntaxHighlighting} prompt={prompt} />
        </Text>
      </button>
    </Row>
  );
};
