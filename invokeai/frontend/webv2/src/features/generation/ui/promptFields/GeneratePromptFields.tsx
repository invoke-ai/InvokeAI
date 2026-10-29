import type { PromptHistoryItem } from '@features/generation/contracts';
import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';
import type { PromptTemplateChange } from '@features/generation/ui/promptFields/PromptTemplatesButton';

import { Stack } from '@chakra-ui/react';
import { getPromptPolicy } from '@features/generation/core/baseGenerationPolicies';
import { sanitizeBatchCount } from '@features/generation/core/batch';
import { flattenPromptTemplateExpansion } from '@features/generation/core/promptTemplates';
import { type GenerateDraft, pickGenerateSettings } from '@features/generation/ui/generateDebounce';
import { flushGenerateDrafts } from '@features/generation/ui/generateDraftRegistry';
import { offerGenerateUndo } from '@features/generation/ui/generateUndo';
import { useGenerateValues, useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { NegativePromptField } from './NegativePromptField';
import { PositivePromptField } from './PositivePromptField';

interface GeneratePromptFieldsProps {
  draft: GenerateDraft;
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

interface GeneratePromptValues {
  negativePrompt: string;
  negativePromptEnabled: boolean;
  negativePromptHeightPx: number;
  positivePrompt: string;
  positivePromptHeightPx: number;
}

/** The flat GenerateSettings field behind each dynamic prompts config key. */
const DYNAMIC_PROMPT_SETTING_KEYS = {
  combinatorial: 'dynamicPromptsCombinatorial',
  enabled: 'dynamicPromptsEnabled',
  maxPrompts: 'dynamicPromptsMaxPrompts',
  resample: 'dynamicPromptsResample',
  sampleSeed: 'dynamicPromptsSampleSeed',
  seedBehaviour: 'dynamicPromptsSeedBehaviour',
} as const satisfies Record<keyof DynamicPromptsConfig, keyof GenerateSettings>;

type DynamicPromptsConfigKey = keyof typeof DYNAMIC_PROMPT_SETTING_KEYS;

const getPromptValues = (values: Record<string, unknown>): GeneratePromptValues => ({
  negativePrompt: typeof values.negativePrompt === 'string' ? values.negativePrompt : '',
  negativePromptEnabled: values.negativePromptEnabled !== false,
  negativePromptHeightPx: typeof values.negativePromptHeightPx === 'number' ? values.negativePromptHeightPx : 56,
  positivePrompt: typeof values.positivePrompt === 'string' ? values.positivePrompt : '',
  positivePromptHeightPx: typeof values.positivePromptHeightPx === 'number' ? values.positivePromptHeightPx : 96,
});

const selectPromptSettings = pickGenerateSettings([
  'dynamicPromptsCombinatorial',
  'dynamicPromptsEnabled',
  'dynamicPromptsMaxPrompts',
  'dynamicPromptsResample',
  'dynamicPromptsSampleSeed',
  'dynamicPromptsSeedBehaviour',
  'expandPromptModelKey',
  'imageToPromptModelKey',
  'loras',
  'promptTemplate',
  'promptTemplateViewMode',
  'seedMode',
]);

const selectBatchCount = (values: Record<string, unknown>): number => sanitizeBatchCount(values.batchCount);

export const GeneratePromptFields = ({
  draft,
  onCommit,
  onCommitImmediate,
  projectId,
  selectedModel,
}: GeneratePromptFieldsProps) => {
  const { t } = useTranslation();
  const ui = useGenerationUi();
  const { showPromptSyntaxHighlighting } = ui.project;
  const promptValues = useGenerateValues(getPromptValues);
  const batchCount = useGenerateValues(selectBatchCount, Object.is);
  const settings = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, selectPromptSettings);
  // Only the negative field's visibility and help text render here; both derive from the model and draft settings.
  const promptPolicy = useExternalStoreSelector(
    draft.subscribe,
    draft.getSnapshot,
    (draftSettings: GenerateSettings) => {
      const { negativeHelpText, negativeVisible } = getPromptPolicy(selectedModel, draftSettings);

      return { negativeHelpText, negativeVisible };
    }
  );

  /**
   * Applies a prompt change that reaches past the prompt text, which Ctrl+Z in the prompt cannot undo, with an Undo
   * toast. Drafts flush first so the Undo restores what the user saw. `isWhole` changes undo all or nothing.
   */
  const commitWithUndo = useCallback(
    (patch: Partial<GenerateSettings>, title: string, isWhole = false) => {
      flushGenerateDrafts();
      const before = ui.settings.readGenerateValues(projectId) ?? {};

      onCommitImmediate(patch);
      offerGenerateUndo({ after: { ...before, ...patch }, before, isWhole, projectId, slot: 'prompt', t, title, ui });
    },
    [onCommitImmediate, projectId, t, ui]
  );

  // The positive prompt also lands in the prompt box as an edit (Ctrl+Z); replacing the negative prompt as well
  // needs the toast's Undo.
  const usePromptHistoryItem = useCallback(
    (prompt: PromptHistoryItem) => {
      if (!promptPolicy.negativeVisible) {
        onCommitImmediate({ positivePrompt: prompt.positivePrompt });
        return;
      }

      const patch = {
        negativePrompt: prompt.negativePrompt ?? '',
        negativePromptEnabled: prompt.negativePrompt ? true : promptValues.negativePromptEnabled,
        positivePrompt: prompt.positivePrompt,
      };

      if (
        patch.negativePrompt === promptValues.negativePrompt &&
        patch.negativePromptEnabled === promptValues.negativePromptEnabled
      ) {
        onCommitImmediate(patch);
      } else {
        commitWithUndo(patch, t('widgets.generate.promptHistoryUsed'));
      }
    },
    [
      commitWithUndo,
      onCommitImmediate,
      promptPolicy.negativeVisible,
      promptValues.negativePrompt,
      promptValues.negativePromptEnabled,
      t,
    ]
  );

  const handlePositivePromptChange = useCallback((positivePrompt: string) => onCommit({ positivePrompt }), [onCommit]);

  const setTemplateViewMode = useCallback(
    (promptTemplateViewMode: boolean) => onCommitImmediate({ promptTemplateViewMode }),
    [onCommitImmediate]
  );

  // Reset view mode when the template changes so prompts do not become unexpectedly read-only. Choosing or clearing
  // one changes what generates without touching the prompt text, so it gets an Undo; keeping it in step with the
  // catalog does not.
  const applyPromptTemplate = useCallback(
    (promptTemplate: PromptTemplateSnapshot | null, change: PromptTemplateChange = 'chosen') => {
      const patch = { promptTemplate, promptTemplateViewMode: false };
      const previous = settings.promptTemplate;

      if (change === 'synced' || promptTemplate?.id === previous?.id) {
        onCommitImmediate(patch);
      } else if (promptTemplate) {
        commitWithUndo(patch, t('widgets.generate.promptTemplateApplied', { name: promptTemplate.name }));
      } else {
        commitWithUndo(patch, t('widgets.generate.promptTemplateCleared', { name: previous?.name ?? '' }));
      }
    },
    [commitWithUndo, onCommitImmediate, settings.promptTemplate, t]
  );

  /** Bake the template into the authored prompt and stop applying it. */
  const flattenPromptTemplate = useCallback(
    (positivePrompt: string) => {
      if (!settings.promptTemplate) {
        onCommitImmediate({ positivePrompt, promptTemplateViewMode: false });
        return;
      }

      commitWithUndo(
        flattenPromptTemplateExpansion({
          authoredNegativePrompt: promptValues.negativePrompt,
          selectedPositivePrompt: positivePrompt,
          template: settings.promptTemplate,
        }),
        t('widgets.generate.promptTemplateMerged', { name: settings.promptTemplate.name }),
        // Half a merge (the negative prompt without its positive) would be a prompt nobody wrote.
        true
      );
    },
    [commitWithUndo, onCommitImmediate, promptValues.negativePrompt, settings.promptTemplate, t]
  );

  const handleDynamicPromptsChange = useCallback(
    (patch: Partial<DynamicPromptsConfig>) =>
      onCommitImmediate(
        Object.fromEntries(
          Object.entries(patch).map(([key, value]) => [
            DYNAMIC_PROMPT_SETTING_KEYS[key as DynamicPromptsConfigKey],
            value,
          ])
        )
      ),
    [onCommitImmediate]
  );

  const dynamicPrompts = useMemo(
    () => ({
      combinatorial: settings.dynamicPromptsCombinatorial,
      enabled: settings.dynamicPromptsEnabled,
      isSeedFixed: settings.seedMode === 'fixed',
      maxPrompts: settings.dynamicPromptsMaxPrompts,
      onChange: handleDynamicPromptsChange,
      resample: settings.dynamicPromptsResample,
      sampleSeed: settings.dynamicPromptsSampleSeed,
      seedBehaviour: settings.dynamicPromptsSeedBehaviour,
    }),
    [
      handleDynamicPromptsChange,
      settings.dynamicPromptsCombinatorial,
      settings.dynamicPromptsEnabled,
      settings.dynamicPromptsMaxPrompts,
      settings.dynamicPromptsResample,
      settings.dynamicPromptsSampleSeed,
      settings.dynamicPromptsSeedBehaviour,
      settings.seedMode,
    ]
  );

  const handlePositivePromptResizeEnd = useCallback(
    (positivePromptHeightPx: number) => onCommitImmediate({ positivePromptHeightPx }),
    [onCommitImmediate]
  );

  const handleNegativePromptEnabledChange = useCallback(
    (negativePromptEnabled: boolean) => onCommitImmediate({ negativePromptEnabled }),
    [onCommitImmediate]
  );

  const handleNegativePromptChange = useCallback((negativePrompt: string) => onCommit({ negativePrompt }), [onCommit]);

  const handleNegativePromptResizeEnd = useCallback(
    (negativePromptHeightPx: number) => onCommitImmediate({ negativePromptHeightPx }),
    [onCommitImmediate]
  );

  const savedPromptModels = useMemo(
    () => ({
      expandPromptModelKey: settings.expandPromptModelKey,
      imageToPromptModelKey: settings.imageToPromptModelKey,
      onChange: onCommitImmediate,
    }),
    [onCommitImmediate, settings.expandPromptModelKey, settings.imageToPromptModelKey]
  );

  return (
    <Stack gap="1" py="2">
      <PositivePromptField
        batchCount={batchCount}
        dynamicPrompts={dynamicPrompts}
        heightPx={promptValues.positivePromptHeightPx}
        value={promptValues.positivePrompt}
        loras={settings.loras}
        projectId={projectId}
        promptTemplate={settings.promptTemplate}
        savedPromptModels={savedPromptModels}
        selectedModel={selectedModel}
        showSyntaxHighlighting={showPromptSyntaxHighlighting}
        isTemplateViewMode={settings.promptTemplateViewMode}
        onApplyPromptTemplate={applyPromptTemplate}
        onChange={handlePositivePromptChange}
        onFlattenPromptTemplate={flattenPromptTemplate}
        onResizeEnd={handlePositivePromptResizeEnd}
        onTemplateViewModeChange={setTemplateViewMode}
        onUsePrompt={usePromptHistoryItem}
      />

      {promptPolicy.negativeVisible ? (
        <NegativePromptField
          heightPx={promptValues.negativePromptHeightPx}
          isEnabled={promptValues.negativePromptEnabled}
          loras={settings.loras}
          projectId={projectId}
          selectedModel={selectedModel}
          helpText={promptPolicy.negativeHelpText}
          isTemplateViewMode={settings.promptTemplateViewMode}
          showSyntaxHighlighting={showPromptSyntaxHighlighting}
          templateNegativePrompt={settings.promptTemplate?.negativePrompt ?? null}
          value={promptValues.negativePrompt}
          onEnabledChange={handleNegativePromptEnabledChange}
          onChange={handleNegativePromptChange}
          onResizeEnd={handleNegativePromptResizeEnd}
          onTemplateViewModeChange={setTemplateViewMode}
        />
      ) : null}
    </Stack>
  );
};
