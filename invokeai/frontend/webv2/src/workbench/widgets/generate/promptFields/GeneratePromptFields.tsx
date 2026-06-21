import type { GenerateModelConfig, GenerateSettings } from '@workbench/generation/types';
import type { PromptHistoryItem } from '@workbench/types';

import { Stack } from '@chakra-ui/react';
import { getPromptPolicy } from '@workbench/generation/baseGenerationPolicies';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';

import { NegativePromptField } from './NegativePromptField';
import { PositivePromptField } from './PositivePromptField';

interface GeneratePromptFieldsProps {
  settings: GenerateSettings;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

export const GeneratePromptFields = ({ onCommit, selectedModel, settings }: GeneratePromptFieldsProps) => {
  const showSyntaxHighlighting = useActiveProjectSelector((project) => project.settings.showPromptSyntaxHighlighting);
  const promptPolicy = getPromptPolicy(selectedModel, settings);
  const usePromptHistoryItem = (prompt: PromptHistoryItem) => {
    onCommit(
      promptPolicy.negativeVisible
        ? {
            negativePrompt: prompt.negativePrompt ?? '',
            negativePromptEnabled: prompt.negativePrompt ? true : settings.negativePromptEnabled,
            positivePrompt: prompt.positivePrompt,
          }
        : { positivePrompt: prompt.positivePrompt }
    );
  };

  return (
    <Stack gap="1" py="2">
      <PositivePromptField
        heightPx={settings.positivePromptHeightPx}
        value={settings.positivePrompt}
        loras={settings.loras}
        selectedModel={selectedModel}
        showSyntaxHighlighting={showSyntaxHighlighting}
        onChange={(positivePrompt) => onCommit({ positivePrompt })}
        onResizeEnd={(positivePromptHeightPx) => onCommit({ positivePromptHeightPx })}
        onUsePrompt={usePromptHistoryItem}
      />
      {promptPolicy.negativeVisible ? (
        <NegativePromptField
          heightPx={settings.negativePromptHeightPx}
          isEnabled={settings.negativePromptEnabled}
          loras={settings.loras}
          selectedModel={selectedModel}
          helpText={promptPolicy.negativeHelpText}
          showSyntaxHighlighting={showSyntaxHighlighting}
          value={settings.negativePrompt}
          onEnabledChange={(negativePromptEnabled) => onCommit({ negativePromptEnabled })}
          onChange={(negativePrompt) => onCommit({ negativePrompt })}
          onResizeEnd={(negativePromptHeightPx) => onCommit({ negativePromptHeightPx })}
        />
      ) : null}
    </Stack>
  );
};
