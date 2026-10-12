import type { GenerateLora, MainModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

import { Stack, Text } from '@chakra-ui/react';
import { NegativePromptField, PositivePromptField } from '@features/generation/components';
import { areProjectPromptDraftsEqual } from '@features/generation/settings';
import { upscaleArchitectureFor } from '@features/upscale/core/settings';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { areLorasEquivalent, areModelsEquivalent } from './upscaleComparators';

/**
 * Compare prompt/LoRA content across reconstructed values to avoid costly rerenders and disturbed autocomplete
 * state.
 */

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
        <Text color="fg.muted" fontSize="xs" textWrap="pretty">
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
        {(upscaleArchitectureFor(model)?.usesNegativePrompt ?? true) && (
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
        )}
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
