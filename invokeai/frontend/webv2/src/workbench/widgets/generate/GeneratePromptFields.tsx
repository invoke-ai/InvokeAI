import type { GenerateModelConfig, GenerateSettings } from '@workbench/generation/types';
import type { ChangeEvent } from 'react';

import { Stack, Textarea } from '@chakra-ui/react';
import { Field } from '@workbench/components/ui';
import { getPromptPolicy } from '@workbench/generation/baseGenerationPolicies';

interface GeneratePromptFieldsProps {
  settings: GenerateSettings;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

interface PromptFieldProps {
  helpText?: string;
  value: string;
  onChange: (value: string) => void;
}

const PositivePromptField = ({ onChange, value }: PromptFieldProps) => (
  <Field label="Prompt">
    <Textarea
      aria-label="Positive prompt"
      minH="6rem"
      resize="vertical"
      size="xs"
      fontFamily="mono"
      value={value}
      onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.currentTarget.value)}
    />
  </Field>
);

const NegativePromptField = ({ helpText, onChange, value }: PromptFieldProps) => (
  <Field label="Negative prompt" helpText={helpText}>
    <Textarea
      aria-label="Negative prompt"
      minH="3.5rem"
      resize="vertical"
      size="xs"
      fontFamily="mono"
      value={value}
      onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.currentTarget.value)}
    />
  </Field>
);

export const GeneratePromptFields = ({ onCommit, selectedModel, settings }: GeneratePromptFieldsProps) => {
  const promptPolicy = getPromptPolicy(selectedModel, settings);

  return (
    <Stack gap="1">
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
