import type { GenerateLora, GenerateModelConfig } from '@workbench/generation/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { HStack, Switch } from '@chakra-ui/react';
import { Field } from '@workbench/components/ui';
import { useRef, useState } from 'react';

import { AddPromptTriggerButton, PromptTriggerPopover } from './PositivePromptActions';
import { PROMPT_ATTENTION_TARGET_PROPS } from './promptAttentionHotkeys';
import { insertPromptText, type PromptTextRange } from './promptFocus';
import { PromptTextarea } from './PromptTextarea';

interface NegativePromptFieldProps {
  heightPx: number;
  helpText?: string;
  isEnabled: boolean;
  loras: GenerateLora[];
  selectedModel: GenerateModelConfig | undefined;
  showSyntaxHighlighting: boolean;
  value: string;
  onChange: (value: string) => void;
  onEnabledChange: (isEnabled: boolean) => void;
  onResizeEnd: (heightPx: number) => void;
}

type PromptTriggerPickerState = {
  anchorRect: { height: number; width: number; x: number; y: number };
  range?: PromptTextRange;
};

export const NegativePromptField = ({
  heightPx,
  helpText,
  isEnabled,
  loras,
  onChange,
  onEnabledChange,
  onResizeEnd,
  selectedModel,
  showSyntaxHighlighting,
  value,
}: NegativePromptFieldProps) => {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [triggerPickerState, setTriggerPickerState] = useState<PromptTriggerPickerState | null>(null);

  const openPromptTriggerPicker = (anchorElement: HTMLElement, range?: PromptTextRange) => {
    const rect = anchorElement.getBoundingClientRect();

    setTriggerPickerState({
      anchorRect: { height: rect.height, width: rect.width, x: rect.x, y: rect.y },
      range,
    });
  };

  const handlePromptKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== '<' || event.altKey || event.ctrlKey || event.metaKey) {
      return;
    }

    event.preventDefault();
    openPromptTriggerPicker(event.currentTarget, {
      end: event.currentTarget.selectionEnd,
      start: event.currentTarget.selectionStart,
    });
  };

  const closePromptTriggerPicker = () => setTriggerPickerState(null);

  const selectPromptTrigger = (trigger: string) => {
    insertPromptText({
      onChange,
      range: triggerPickerState?.range,
      textarea: textareaRef.current,
      text: trigger,
      value,
    });
    closePromptTriggerPicker();
  };

  return (
    <Field
      label="Negative prompt"
      labelEnd={
        <HStack gap="0.5">
          {isEnabled ? (
            <AddPromptTriggerButton
              isOpen={triggerPickerState !== null}
              onOpenPromptTriggerPicker={(anchorElement) => openPromptTriggerPicker(anchorElement)}
            />
          ) : null}
          <Switch.Root checked={isEnabled} size="sm" onCheckedChange={(event) => onEnabledChange(event.checked)}>
            <Switch.HiddenInput />
            <Switch.Control _checked={{ bg: 'accent.solid' }}>
              <Switch.Thumb />
            </Switch.Control>
            <Switch.Label srOnly>Enable negative prompt</Switch.Label>
          </Switch.Root>
        </HStack>
      }
      helpText={isEnabled ? helpText : undefined}
    >
      {isEnabled ? (
        <>
          <PromptTextarea
            {...PROMPT_ATTENTION_TARGET_PROPS}
            aria-label="Negative prompt"
            defaultHeightPx={heightPx}
            maxHeightPx={240}
            minHeightPx={56}
            resizeHandleAriaLabel="Resize negative prompt"
            size="xs"
            fontFamily="mono"
            showSyntaxHighlighting={showSyntaxHighlighting}
            textareaRef={(element) => {
              textareaRef.current = element;
            }}
            value={value}
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.currentTarget.value)}
            onKeyDown={handlePromptKeyDown}
            onResizeEnd={onResizeEnd}
          />
          <PromptTriggerPopover
            loras={loras}
            open={triggerPickerState !== null}
            positioning={{ getAnchorRect: () => triggerPickerState?.anchorRect ?? null }}
            selectedModel={selectedModel}
            onClose={closePromptTriggerPicker}
            onSelect={selectPromptTrigger}
          />
        </>
      ) : null}
    </Field>
  );
};
