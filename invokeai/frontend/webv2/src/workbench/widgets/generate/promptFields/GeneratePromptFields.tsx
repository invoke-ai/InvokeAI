import type { GenerateModelConfig, GenerateSettings } from '@workbench/generation/types';
import type { PromptHistoryItem } from '@workbench/types';

import { Stack } from '@chakra-ui/react';
import { getPromptPolicy } from '@workbench/generation/baseGenerationPolicies';
import { useActiveProjectSelector, useWidgetValuesSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';

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

const getPromptValues = (values: Record<string, unknown>): GeneratePromptValues => ({
  negativePrompt: typeof values.negativePrompt === 'string' ? values.negativePrompt : '',
  negativePromptEnabled: values.negativePromptEnabled !== false,
  negativePromptHeightPx: typeof values.negativePromptHeightPx === 'number' ? values.negativePromptHeightPx : 56,
  positivePrompt: typeof values.positivePrompt === 'string' ? values.positivePrompt : '',
  positivePromptHeightPx: typeof values.positivePromptHeightPx === 'number' ? values.positivePromptHeightPx : 96,
});

const arePromptValuesEqual = (left: GeneratePromptValues, right: GeneratePromptValues): boolean =>
  left.positivePrompt === right.positivePrompt &&
  left.negativePrompt === right.negativePrompt &&
  left.positivePromptHeightPx === right.positivePromptHeightPx &&
  left.negativePromptHeightPx === right.negativePromptHeightPx &&
  left.negativePromptEnabled === right.negativePromptEnabled;

export const GeneratePromptFields = ({
  onCommit,
  onCommitImmediate,
  projectId,
  selectedModel,
  settings,
}: GeneratePromptFieldsProps) => {
  const showSyntaxHighlighting = useActiveProjectSelector((project) => project.settings.showPromptSyntaxHighlighting);
  const promptValues = useWidgetValuesSelector('generate', getPromptValues, arePromptValuesEqual);
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
        heightPx={promptValues.positivePromptHeightPx}
        value={promptValues.positivePrompt}
        loras={settings.loras}
        projectId={projectId}
        selectedModel={selectedModel}
        showSyntaxHighlighting={showSyntaxHighlighting}
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
          showSyntaxHighlighting={showSyntaxHighlighting}
          value={promptValues.negativePrompt}
          onEnabledChange={handleNegativePromptEnabledChange}
          onChange={handleNegativePromptChange}
          onResizeEnd={handleNegativePromptResizeEnd}
        />
      ) : null}
    </Stack>
  );
};
