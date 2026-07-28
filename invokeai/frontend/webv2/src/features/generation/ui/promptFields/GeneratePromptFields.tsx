import type { PromptHistoryItem } from '@features/generation/contracts';
import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Stack } from '@chakra-ui/react';
import { getPromptPolicy } from '@features/generation/core/baseGenerationPolicies';
import { sanitizeBatchCount } from '@features/generation/core/batch';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { useCallback, useMemo } from 'react';

import { NegativePromptField } from './NegativePromptField';
import { PositivePromptField } from './PositivePromptField';

interface GeneratePromptFieldsProps {
  settings: GenerateSettings;
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
  maxPrompts: 'dynamicPromptsMaxPrompts',
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

export const GeneratePromptFields = ({
  onCommit,
  onCommitImmediate,
  projectId,
  selectedModel,
  settings,
}: GeneratePromptFieldsProps) => {
  const { generateValues, showPromptSyntaxHighlighting } = useGenerationUi().project;
  const promptValues = getPromptValues(generateValues);
  const promptPolicy = getPromptPolicy(selectedModel, settings);

  const usePromptHistoryItem = useCallback(
    (prompt: PromptHistoryItem) => {
      onCommitImmediate(
        promptPolicy.negativeVisible
          ? {
              negativePrompt: prompt.negativePrompt ?? '',
              negativePromptEnabled: prompt.negativePrompt ? true : promptValues.negativePromptEnabled,
              positivePrompt: prompt.positivePrompt,
            }
          : { positivePrompt: prompt.positivePrompt }
      );
    },
    [onCommitImmediate, promptPolicy.negativeVisible, promptValues.negativePromptEnabled]
  );

  const handlePositivePromptChange = useCallback((positivePrompt: string) => onCommit({ positivePrompt }), [onCommit]);

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
      maxPrompts: settings.dynamicPromptsMaxPrompts,
      onChange: handleDynamicPromptsChange,
      sampleSeed: settings.dynamicPromptsSampleSeed,
      seedBehaviour: settings.dynamicPromptsSeedBehaviour,
    }),
    [
      handleDynamicPromptsChange,
      settings.dynamicPromptsCombinatorial,
      settings.dynamicPromptsMaxPrompts,
      settings.dynamicPromptsSampleSeed,
      settings.dynamicPromptsSeedBehaviour,
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

  return (
    <Stack gap="1" py="2">
      <PositivePromptField
        batchCount={sanitizeBatchCount(generateValues.batchCount)}
        dynamicPrompts={dynamicPrompts}
        heightPx={promptValues.positivePromptHeightPx}
        value={promptValues.positivePrompt}
        loras={settings.loras}
        projectId={projectId}
        selectedModel={selectedModel}
        showSyntaxHighlighting={showPromptSyntaxHighlighting}
        onChange={handlePositivePromptChange}
        onResizeEnd={handlePositivePromptResizeEnd}
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
          showSyntaxHighlighting={showPromptSyntaxHighlighting}
          value={promptValues.negativePrompt}
          onEnabledChange={handleNegativePromptEnabledChange}
          onChange={handleNegativePromptChange}
          onResizeEnd={handleNegativePromptResizeEnd}
        />
      ) : null}
    </Stack>
  );
};
