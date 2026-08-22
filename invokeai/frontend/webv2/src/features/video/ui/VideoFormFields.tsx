import type { GenerateLora, MainModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { VideoWidgetValues } from '@features/video/core/types';

import { Stack, Text } from '@chakra-ui/react';
import { NegativePromptField, PositivePromptField } from '@features/generation/components';
import { areProjectPromptDraftsEqual } from '@features/generation/settings';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { areVideoLorasEquivalent, areVideoModelsEquivalent } from './videoComparators';

/**
 * The Video widget's prompt block. Shares the project-wide prompt draft with
 * Generate (upscale-style), so a prompt travels with the user between panels.
 * Memoised against content: the widget re-derives `values` on every patch and
 * the prompt editors carry autocomplete state a needless remount would disturb.
 */
export const VideoPromptFields = memo(
  function VideoPromptFields({
    loras,
    model,
    negativeHelpText,
    negativePromptHeightPx,
    negativeVisible,
    onPatchPromptDraft,
    onPatchValues,
    positivePromptHeightPx,
    projectId,
    promptDraft,
    showSyntaxHighlighting,
  }: {
    loras: GenerateLora[];
    model: MainModelConfig | null;
    negativeHelpText?: string;
    negativePromptHeightPx: number;
    /** From the video prompt policy: MiniMax H3 has no negative conditioning at all. */
    negativeVisible: boolean;
    onPatchPromptDraft: (patch: ProjectPromptDraftPatch) => void;
    onPatchValues: (patch: Partial<VideoWidgetValues>) => void;
    positivePromptHeightPx: number;
    projectId: string;
    promptDraft: ProjectPromptDraft;
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
          {t('widgets.video.sharedPromptDescription')}
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
        {negativeVisible ? (
          <NegativePromptField
            heightPx={negativePromptHeightPx}
            helpText={negativeHelpText}
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
        ) : null}
      </Stack>
    );
  },
  (previous, next) =>
    previous.negativePromptHeightPx === next.negativePromptHeightPx &&
    previous.negativeVisible === next.negativeVisible &&
    previous.negativeHelpText === next.negativeHelpText &&
    previous.onPatchPromptDraft === next.onPatchPromptDraft &&
    previous.onPatchValues === next.onPatchValues &&
    previous.positivePromptHeightPx === next.positivePromptHeightPx &&
    previous.projectId === next.projectId &&
    areProjectPromptDraftsEqual(previous.promptDraft, next.promptDraft) &&
    previous.showSyntaxHighlighting === next.showSyntaxHighlighting &&
    areVideoModelsEquivalent(previous.model, next.model) &&
    areVideoLorasEquivalent(previous.loras, next.loras)
);
