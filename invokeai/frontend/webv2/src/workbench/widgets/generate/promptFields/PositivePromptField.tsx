import type { ChangeEvent } from 'react';

import { Field, ResizableTextarea } from '@workbench/components/ui';

import { registerPositivePromptElement } from './promptFocus';

interface PositivePromptFieldProps {
  value: string;
  onChange: (value: string) => void;
}

export const PositivePromptField = ({ onChange, value }: PositivePromptFieldProps) => (
  <Field label="Prompt">
    <ResizableTextarea
      aria-label="Positive prompt"
      defaultHeightPx={96}
      maxHeightPx={360}
      minHeightPx={96}
      resizeHandleAriaLabel="Resize positive prompt"
      size="xs"
      fontFamily="mono"
      textareaRef={registerPositivePromptElement}
      value={value}
      onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.currentTarget.value)}
    />
  </Field>
);
