import type { ChangeEvent } from 'react';

import { Field, ResizableTextarea } from '@workbench/components/ui';

interface NegativePromptFieldProps {
  helpText?: string;
  value: string;
  onChange: (value: string) => void;
}

export const NegativePromptField = ({ helpText, onChange, value }: NegativePromptFieldProps) => (
  <Field label="Negative prompt" helpText={helpText}>
    <ResizableTextarea
      aria-label="Negative prompt"
      defaultHeightPx={56}
      maxHeightPx={240}
      minHeightPx={56}
      resizeHandleAriaLabel="Resize negative prompt"
      size="xs"
      fontFamily="mono"
      value={value}
      onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.currentTarget.value)}
    />
  </Field>
);
