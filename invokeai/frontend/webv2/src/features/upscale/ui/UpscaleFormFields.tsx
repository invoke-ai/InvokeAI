import type { PromptTemplateChange } from '@features/generation/components';
import type { GenerateLora, MainModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type {
  DynamicPromptsConfig,
  ProjectPromptDraft,
  ProjectPromptDraftPatch,
  PromptTemplateSnapshot,
} from '@features/generation/settings';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

import { Stack, Text } from '@chakra-ui/react';
import { NegativePromptField, PositivePromptField } from '@features/generation/components';
import { getGeneratePromptUndoKey } from '@features/generation/react';
import {
  areProjectPromptDraftsEqual,
  flattenPromptTemplateExpansion,
  getDynamicPromptsConfig,
  getDynamicPromptsSettingsPatch,
} from '@features/generation/settings';
import { upscaleArchitectureFor } from '@features/upscale/core/settings';
import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { areLorasEquivalent, areModelsEquivalent } from './upscaleComparators';
import { useUpscaleUiActions } from './UpscaleUiContext';

/** The draft keys a change sets to something other than what they held. */
const getChangedKeys = (before: ProjectPromptDraft, written: ProjectPromptDraftPatch): (keyof ProjectPromptDraft)[] =>
  (Object.keys(written) as (keyof ProjectPromptDraft)[]).filter(
    (key) => !areJsonValuesStructurallyEqual(before[key], written[key])
  );

/**
 * Compare prompt/LoRA content across reconstructed values to avoid costly rerenders and disturbed autocomplete
 * state.
 */

export const UpscalePromptFields = memo(
  function UpscalePromptFields({
    batchCount,
    isSeedFixed,
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
    batchCount: number;
    isSeedFixed: boolean;
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

    /**
     * Applies a prompt change that reaches past the prompt text, which Ctrl+Z in the prompt cannot undo, with an Undo
     * toast. The prompt is Generate's too, so this Undo takes Generate's prompt slot. `isWhole` changes undo all or
     * nothing.
     */
    const patchWithUndo = useCallback(
      (written: ProjectPromptDraftPatch, title: string, isWhole = false) => {
        // Typing still inside a draft's debounce lands first, so the Undo restores what the user saw.
        flushWorkbenchDrafts();
        const before = readPromptDraft();
        const changedKeys = getChangedKeys(before, written);

        onPatchPromptDraft(written);

        if (changedKeys.length === 0) {
          return;
        }

        notifications.undoable(title, {
          onUndo: () => {
            flushWorkbenchDrafts();
            const current = readPromptDraft();
            // Each key still holding what the change wrote goes back; a key edited since keeps the edit.
            const stillApplied = changedKeys.filter((key) =>
              areJsonValuesStructurallyEqual(current[key], written[key])
            );

            if (stillApplied.length > 0 && (!isWhole || stillApplied.length === changedKeys.length)) {
              onPatchPromptDraft(Object.fromEntries(stillApplied.map((key) => [key, before[key]])));
            } else {
              notifications.info(t('widgets.generate.undoStale'));
            }
          },
          supersedes: getGeneratePromptUndoKey(projectId),
        });
      },
      [notifications, onPatchPromptDraft, projectId, readPromptDraft, t]
    );

    // The positive prompt also lands in the prompt box as an edit (Ctrl+Z); replacing the negative prompt as well
    // needs the toast's Undo.
    const handleUsePrompt = useCallback(
      (prompt: PromptHistoryItem) => {
        // A hidden negative prompt is not this model's to replace.
        if (negativePromptUsage === 'never') {
          onPatchPromptDraft({ positivePrompt: prompt.positivePrompt });
          return;
        }

        const before = readPromptDraft();
        const written: ProjectPromptDraftPatch = {
          negativePrompt: prompt.negativePrompt ?? '',
          negativePromptEnabled: prompt.negativePrompt ? true : before.negativePromptEnabled,
          positivePrompt: prompt.positivePrompt,
        };

        if (
          written.negativePrompt === before.negativePrompt &&
          written.negativePromptEnabled === before.negativePromptEnabled
        ) {
          onPatchPromptDraft(written);
        } else {
          patchWithUndo(written, t('widgets.generate.promptHistoryUsed'));
        }
      },
      [negativePromptUsage, onPatchPromptDraft, patchWithUndo, readPromptDraft, t]
    );

    const { promptTemplate } = promptDraft;

    // View mode resets when the template changes so prompts do not become unexpectedly read-only. Choosing or clearing
    // one changes what generates without touching the prompt text, so it gets an Undo; keeping it in step with the
    // catalog does not.
    const applyPromptTemplate = useCallback(
      (template: PromptTemplateSnapshot | null, change: PromptTemplateChange) => {
        const patch: ProjectPromptDraftPatch = { promptTemplate: template, promptTemplateViewMode: false };

        if (change === 'synced' || template?.id === promptTemplate?.id) {
          onPatchPromptDraft(patch);
        } else if (template) {
          patchWithUndo(patch, t('widgets.generate.promptTemplateApplied', { name: template.name }));
        } else {
          patchWithUndo(patch, t('widgets.generate.promptTemplateCleared', { name: promptTemplate?.name ?? '' }));
        }
      },
      [onPatchPromptDraft, patchWithUndo, promptTemplate, t]
    );

    /** Bake the template into the authored prompt and stop applying it. */
    const flattenPromptTemplate = useCallback(
      (positivePrompt: string) => {
        if (!promptTemplate) {
          onPatchPromptDraft({ positivePrompt, promptTemplateViewMode: false });
          return;
        }

        flushWorkbenchDrafts();
        patchWithUndo(
          flattenPromptTemplateExpansion({
            authoredNegativePrompt: readPromptDraft().negativePrompt,
            selectedPositivePrompt: positivePrompt,
            template: promptTemplate,
          }),
          t('widgets.generate.promptTemplateMerged', { name: promptTemplate.name }),
          // Half a merge (the negative prompt without its positive) would be a prompt nobody wrote.
          true
        );
      },
      [onPatchPromptDraft, patchWithUndo, promptTemplate, readPromptDraft, t]
    );

    const setTemplateViewMode = useCallback(
      (promptTemplateViewMode: boolean) => onPatchPromptDraft({ promptTemplateViewMode }),
      [onPatchPromptDraft]
    );

    const handleDynamicPromptsChange = useCallback(
      (patch: Partial<DynamicPromptsConfig>) => onPatchPromptDraft(getDynamicPromptsSettingsPatch(patch)),
      [onPatchPromptDraft]
    );
    const {
      dynamicPromptsCombinatorial,
      dynamicPromptsEnabled,
      dynamicPromptsMaxPrompts,
      dynamicPromptsResample,
      dynamicPromptsSampleSeed,
      dynamicPromptsSeedBehaviour,
    } = promptDraft;
    // The settings are Generate's: the shared prompt expands the same way wherever it runs.
    const dynamicPrompts = useMemo(
      () => ({
        ...getDynamicPromptsConfig({
          dynamicPromptsCombinatorial,
          dynamicPromptsEnabled,
          dynamicPromptsMaxPrompts,
          dynamicPromptsResample,
          dynamicPromptsSampleSeed,
          dynamicPromptsSeedBehaviour,
        }),
        isSeedFixed,
        onChange: handleDynamicPromptsChange,
      }),
      [
        dynamicPromptsCombinatorial,
        dynamicPromptsEnabled,
        dynamicPromptsMaxPrompts,
        dynamicPromptsResample,
        dynamicPromptsSampleSeed,
        dynamicPromptsSeedBehaviour,
        handleDynamicPromptsChange,
        isSeedFixed,
      ]
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
      <Stack gap="2.5">
        <Text color="fg.muted" fontSize="xs" textWrap="pretty">
          {t('widgets.upscale.sharedPromptDescription')}
        </Text>
        <PositivePromptField
          batchCount={batchCount}
          dynamicPrompts={dynamicPrompts}
          focusOwner="upscale"
          heightPx={positivePromptHeightPx}
          isTemplateViewMode={promptDraft.promptTemplateViewMode}
          loras={loras}
          projectId={projectId}
          promptTemplate={promptTemplate}
          selectedModel={model ?? undefined}
          showSyntaxHighlighting={showSyntaxHighlighting}
          takesPendingTemplateDraft={false}
          value={promptDraft.positivePrompt}
          onApplyPromptTemplate={applyPromptTemplate}
          onChange={handlePositiveChange}
          onFlattenPromptTemplate={flattenPromptTemplate}
          onResizeEnd={handlePositiveResizeEnd}
          onTemplateViewModeChange={setTemplateViewMode}
          onUsePrompt={handleUsePrompt}
        />
        {negativePromptUsage !== 'never' && (
          <NegativePromptField
            heightPx={negativePromptHeightPx}
            helpText={negativePromptUsage === 'cfg-gated' ? t('widgets.upscale.negativePromptCfgGated') : undefined}
            isEnabled={promptDraft.negativePromptEnabled}
            isTemplateViewMode={promptDraft.promptTemplateViewMode}
            loras={loras}
            projectId={projectId}
            selectedModel={model ?? undefined}
            showSyntaxHighlighting={showSyntaxHighlighting}
            templateNegativePrompt={promptTemplate?.negativePrompt ?? null}
            value={promptDraft.negativePrompt}
            onChange={handleNegativeChange}
            onEnabledChange={handleNegativeEnabledChange}
            onResizeEnd={handleNegativeResizeEnd}
            onTemplateViewModeChange={setTemplateViewMode}
          />
        )}
      </Stack>
    );
  },
  (previous, next) =>
    previous.batchCount === next.batchCount &&
    previous.isSeedFixed === next.isSeedFixed &&
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
