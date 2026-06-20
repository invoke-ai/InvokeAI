import type { GenerateModelConfig, GenerateSettings } from '@workbench/generation/types';

import { Stack } from '@chakra-ui/react';
import { getPromptPolicy } from '@workbench/generation/baseGenerationPolicies';

import { NegativePromptField } from './NegativePromptField';
import { PositivePromptField } from './PositivePromptField';

interface GeneratePromptFieldsProps {
  settings: GenerateSettings;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

export const GeneratePromptFields = ({ onCommit, selectedModel, settings }: GeneratePromptFieldsProps) => {
  const promptPolicy = getPromptPolicy(selectedModel, settings);

  return (
    <Stack gap="1" py="2">
      <PositivePromptField
        heightPx={settings.positivePromptHeightPx}
        value={settings.positivePrompt}
        onChange={(positivePrompt) => onCommit({ positivePrompt })}
        onResizeEnd={(positivePromptHeightPx) => onCommit({ positivePromptHeightPx })}
      />
      {promptPolicy.negativeVisible ? (
        <NegativePromptField
          heightPx={settings.negativePromptHeightPx}
          isEnabled={settings.negativePromptEnabled}
          helpText={promptPolicy.negativeHelpText}
          value={settings.negativePrompt}
          onEnabledChange={(negativePromptEnabled) => onCommit({ negativePromptEnabled })}
          onChange={(negativePrompt) => onCommit({ negativePrompt })}
          onResizeEnd={(negativePromptHeightPx) => onCommit({ negativePromptHeightPx })}
        />
      ) : null}
    </Stack>
  );
};
