import { Stack, Textarea } from '@chakra-ui/react';
import type { ChangeEvent } from 'react';

import { Field } from '@workbench/components/ui/Field';
import type { GenerateSettings } from '@workbench/generation/types';

interface GeneratePromptFieldsProps {
  settings: GenerateSettings;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

interface PromptFieldProps {
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

const NegativePromptField = ({ onChange, value }: PromptFieldProps) => (
  <Field label="Negative prompt">
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

export const GeneratePromptFields = ({ onCommit, settings }: GeneratePromptFieldsProps) => {
  return (
    <Stack gap="1">
      <PositivePromptField
        value={settings.positivePrompt}
        onChange={(positivePrompt) => onCommit({ positivePrompt })}
      />
      <NegativePromptField
        value={settings.negativePrompt}
        onChange={(negativePrompt) => onCommit({ negativePrompt })}
      />
    </Stack>
  );
};
