import type { GenerateLora, MainModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

import { Stack, Text } from '@chakra-ui/react';
import { NegativePromptField, PositivePromptField } from '@features/generation/components';
import { getGeneratePromptUndoKey } from '@features/generation/react';
import { areProjectPromptDraftsEqual } from '@features/generation/settings';
import { upscaleArchitectureFor } from '@features/upscale/core/settings';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { areLorasEquivalent, areModelsEquivalent } from './upscaleComparators';
import { useUpscaleUiActions } from './UpscaleUiContext';

const PROMPT_DRAFT_KEYS = ['negativePrompt', 'negativePromptEnabled', 'positivePrompt'] as const;

/** The earlier value of each key a recall changed, where it still holds what the recall wrote; edits since stay. */
const getRecallRevert = (
  before: ProjectPromptDraft,
  written: ProjectPromptDraft,
  current: ProjectPromptDraft
): ProjectPromptDraftPatch =>
  Object.fromEntries(
    PROMPT_DRAFT_KEYS.filter((key) => before[key] !== written[key] && current[key] === written[key]).map((key) => [
      key,
      before[key],
    ])
  );

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
    const { notifications, readPromptDraft } = useUpscaleUiActions();
    const negativePromptUsage = upscaleArchitectureFor(model)?.negativePrompt ?? 'always';
    // The positive prompt also lands in the prompt box as an edit (Ctrl+Z); replacing the negative prompt as well
    // needs the toast's Undo. The prompt is Generate's too, so this Undo takes Generate's prompt slot.
    const handleUsePrompt = useCallback(
      (prompt: PromptHistoryItem) => {
        // A hidden negative prompt is not this model's to replace.
        if (negativePromptUsage === 'never') {
          onPatchPromptDraft({ positivePrompt: prompt.positivePrompt });
          return;
        }

        const before = readPromptDraft();
        const written: ProjectPromptDraft = {
          negativePrompt: prompt.negativePrompt ?? '',
          negativePromptEnabled: prompt.negativePrompt ? true : before.negativePromptEnabled,
          positivePrompt: prompt.positivePrompt,
        };

        onPatchPromptDraft(written);

        if (
          written.negativePrompt === before.negativePrompt &&
          written.negativePromptEnabled === before.negativePromptEnabled
        ) {
          return;
        }

        notifications.undoable(t('widgets.generate.promptHistoryUsed'), {
          onUndo: () => {
            // Typing still inside a draft's debounce counts as an edit since, so it lands before the comparison.
            flushWorkbenchDrafts();
            const revert = getRecallRevert(before, written, readPromptDraft());

            if (Object.keys(revert).length > 0) {
              onPatchPromptDraft(revert);
            } else {
              notifications.info(t('widgets.generate.undoStale'));
            }
          },
          // Upscale shares Generate's prompt, so a recall here and one there undo in one slot.
          supersedes: getGeneratePromptUndoKey(projectId),
        });
      },
      [negativePromptUsage, notifications, onPatchPromptDraft, projectId, readPromptDraft, t]
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
        {negativePromptUsage !== 'never' && (
          <NegativePromptField
            heightPx={negativePromptHeightPx}
            helpText={negativePromptUsage === 'cfg-gated' ? t('widgets.upscale.negativePromptCfgGated') : undefined}
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
