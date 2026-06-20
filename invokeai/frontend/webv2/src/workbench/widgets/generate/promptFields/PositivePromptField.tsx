import type { ChangeEvent } from 'react';

import { Field, ResizableTextarea } from '@workbench/components/ui';

import { registerPositivePromptElement } from './promptFocus';

interface PositivePromptFieldProps {
  heightPx: number;
  value: string;
  onChange: (value: string) => void;
  onResizeEnd: (heightPx: number) => void;
}

export const PositivePromptField = ({ heightPx, onChange, onResizeEnd, value }: PositivePromptFieldProps) => (
  <Field label="Prompt">
    <ResizableTextarea
      aria-label="Positive prompt"
      defaultHeightPx={heightPx}
      maxHeightPx={360}
      minHeightPx={96}
      resizeHandleAriaLabel="Resize positive prompt"
      size="xs"
      fontFamily="mono"
      textareaRef={registerPositivePromptElement}
      value={value}
      onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.currentTarget.value)}
      onResizeEnd={onResizeEnd}
    />
  </Field>
);
