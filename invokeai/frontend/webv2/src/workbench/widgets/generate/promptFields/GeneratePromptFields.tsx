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
    <Stack gap="1" pt="2">
      <PositivePromptField
        value={settings.positivePrompt}
        onChange={(positivePrompt) => onCommit({ positivePrompt })}
      />
      {promptPolicy.negativeVisible ? (
        <NegativePromptField
          helpText={promptPolicy.negativeHelpText}
          value={settings.negativePrompt}
          onChange={(negativePrompt) => onCommit({ negativePrompt })}
        />
      ) : null}
    </Stack>
  );
};
