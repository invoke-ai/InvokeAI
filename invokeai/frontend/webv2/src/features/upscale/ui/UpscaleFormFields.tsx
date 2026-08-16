import type { GenerateLora, MainModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

import { HStack, NumberInput, Stack, Switch, Text } from '@chakra-ui/react';
import { NegativePromptField, PositivePromptField } from '@features/generation/components';
import { areProjectPromptDraftsEqual } from '@features/generation/settings';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Trash2Icon } from 'lucide-react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { areLorasEquivalent, areModelsEquivalent } from './upscaleComparators';

/**
 * The Upscale widget's prompt and LoRA controls.
 *
 * Both are memoised against content rather than identity (see
 * `upscaleComparators`), because the widget re-derives `values` on every patch
 * and these are the most expensive sections to re-render -- the prompt editors
 * carry autocomplete state that a needless remount would disturb.
 */

const SWITCH_CHECKED_PROPS = { bg: 'accent.solid' };

export const UpscalePromptFields = memo(
  function UpscalePromptFields({
    loras,
    model,
    negativePromptHeightPx,
    onPatchPromptDraft,
    onPatchValues,
    positivePromptHeightPx,
    promptDraft,
    projectId,
    showSyntaxHighlighting,
  }: {
    loras: GenerateLora[];
    model: MainModelConfig | null;
    negativePromptHeightPx: number;
    onPatchPromptDraft: (patch: ProjectPromptDraftPatch) => void;
    onPatchValues: (patch: Partial<UpscaleWidgetValues>) => void;
    positivePromptHeightPx: number;
    promptDraft: ProjectPromptDraft;
    projectId: string;
    showSyntaxHighlighting: boolean;
  }) {
    const { t } = useTranslation();
    const handleUsePrompt = useCallback(
      (prompt: PromptHistoryItem) =>
        onPatchPromptDraft({
          negativePrompt: prompt.negativePrompt ?? '',
          negativePromptEnabled: prompt.negativePrompt ? true : promptDraft.negativePromptEnabled,
          positivePrompt: prompt.positivePrompt,
        }),
      [onPatchPromptDraft, promptDraft.negativePromptEnabled]
    );
    const handlePositiveChange = useCallback(
      (positivePrompt: string) => onPatchPromptDraft({ positivePrompt }),
      [onPatchPromptDraft]
    );
    const handleNegativeChange = useCallback(
      (negativePrompt: string) => onPatchPromptDraft({ negativePrompt }),
      [onPatchPromptDraft]
    );
    const handleNegativeEnabledChange = useCallback(
      (negativePromptEnabled: boolean) => onPatchPromptDraft({ negativePromptEnabled }),
      [onPatchPromptDraft]
    );
    const handlePositiveResizeEnd = useCallback(
      (positivePromptHeight: number) => onPatchValues({ positivePromptHeightPx: positivePromptHeight }),
      [onPatchValues]
    );
    const handleNegativeResizeEnd = useCallback(
      (negativePromptHeight: number) => onPatchValues({ negativePromptHeightPx: negativePromptHeight }),
      [onPatchValues]
    );

    return (
      <Stack gap="2" p="2">
        <Text color="fg.muted" fontSize="2xs" textWrap="pretty">
          {t('widgets.upscale.sharedPromptDescription')}
        </Text>
        <PositivePromptField
          heightPx={positivePromptHeightPx}
          loras={loras}
          projectId={projectId}
          selectedModel={model ?? undefined}
          showSyntaxHighlighting={showSyntaxHighlighting}
          value={promptDraft.positivePrompt}
          onChange={handlePositiveChange}
          onResizeEnd={handlePositiveResizeEnd}
          onUsePrompt={handleUsePrompt}
        />
        <NegativePromptField
          heightPx={negativePromptHeightPx}
          isEnabled={promptDraft.negativePromptEnabled}
          loras={loras}
          projectId={projectId}
          selectedModel={model ?? undefined}
          showSyntaxHighlighting={showSyntaxHighlighting}
          value={promptDraft.negativePrompt}
          onChange={handleNegativeChange}
          onEnabledChange={handleNegativeEnabledChange}
          onResizeEnd={handleNegativeResizeEnd}
        />
      </Stack>
    );
  },
  (previous, next) =>
    previous.negativePromptHeightPx === next.negativePromptHeightPx &&
    previous.onPatchPromptDraft === next.onPatchPromptDraft &&
    previous.onPatchValues === next.onPatchValues &&
    previous.positivePromptHeightPx === next.positivePromptHeightPx &&
    previous.projectId === next.projectId &&
    areProjectPromptDraftsEqual(previous.promptDraft, next.promptDraft) &&
    previous.showSyntaxHighlighting === next.showSyntaxHighlighting &&
    areModelsEquivalent(previous.model, next.model) &&
    areLorasEquivalent(previous.loras, next.loras)
);

/**
 * One LoRA row. Split out so editing a single weight re-renders that row rather
 * than the whole list; the handlers bind the key here instead of at the call
 * site, where they would be new closures per row per render.
 */
export const UpscaleLoraRow = memo(function UpscaleLoraRow({
  lora,
  onRemove,
  onUpdate,
}: {
  lora: GenerateLora;
  onRemove: (key: string) => void;
  onUpdate: (key: string, update: Partial<GenerateLora>) => void;
}) {
  const { t } = useTranslation();
  const modelKey = lora.model.key;
  const handleToggle = useCallback(
    (details: { checked: boolean }) => onUpdate(modelKey, { isEnabled: details.checked }),
    [modelKey, onUpdate]
  );
  const handleWeightChange = useCallback(
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        onUpdate(modelKey, { weight: valueAsNumber });
      }
    },
    [modelKey, onUpdate]
  );
  const handleRemove = useCallback(() => onRemove(modelKey), [modelKey, onRemove]);

  return (
    <HStack bg="bg.subtle" gap="2" p="2" rounded="md">
      <Switch.Root aria-label={lora.model.name} checked={lora.isEnabled} size="sm" onCheckedChange={handleToggle}>
        <Switch.HiddenInput />
        <Switch.Control _checked={SWITCH_CHECKED_PROPS}>
          <Switch.Thumb />
        </Switch.Control>
      </Switch.Root>
      <MiddleTruncate flex="1" fontSize="xs" minW="0" text={lora.model.name} />
      <NumberInput.Root
        max={10}
        min={-10}
        size="xs"
        step={0.05}
        value={String(lora.weight)}
        w="20"
        onValueChange={handleWeightChange}
      >
        <NumberInput.Input aria-label={t('widgets.upscale.loraWeight', { name: lora.model.name })} />
      </NumberInput.Root>
      <IconButton
        aria-label={t('widgets.upscale.removeLora', { name: lora.model.name })}
        size="xs"
        variant="ghost"
        onClick={handleRemove}
      >
        <Trash2Icon />
      </IconButton>
    </HStack>
  );
});
