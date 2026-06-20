import type { ChangeEvent } from 'react';

import { Switch } from '@chakra-ui/react';
import { Field, ResizableTextarea } from '@workbench/components/ui';

interface NegativePromptFieldProps {
  heightPx: number;
  helpText?: string;
  isEnabled: boolean;
  value: string;
  onChange: (value: string) => void;
  onEnabledChange: (isEnabled: boolean) => void;
  onResizeEnd: (heightPx: number) => void;
}

export const NegativePromptField = ({
  heightPx,
  helpText,
  isEnabled,
  onChange,
  onEnabledChange,
  onResizeEnd,
  value,
}: NegativePromptFieldProps) => (
  <Field
    label="Negative prompt"
    labelEnd={
      <Switch.Root checked={isEnabled} size="sm" onCheckedChange={(event) => onEnabledChange(event.checked)}>
        <Switch.HiddenInput />
        <Switch.Control _checked={{ bg: 'accent.solid' }}>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label srOnly>Enable negative prompt</Switch.Label>
      </Switch.Root>
    }
    helpText={isEnabled ? helpText : undefined}
  >
    {isEnabled ? (
      <ResizableTextarea
        aria-label="Negative prompt"
        defaultHeightPx={heightPx}
        maxHeightPx={240}
        minHeightPx={56}
        resizeHandleAriaLabel="Resize negative prompt"
        size="xs"
        fontFamily="mono"
        value={value}
        onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.currentTarget.value)}
        onResizeEnd={onResizeEnd}
      />
    ) : null}
  </Field>
);
