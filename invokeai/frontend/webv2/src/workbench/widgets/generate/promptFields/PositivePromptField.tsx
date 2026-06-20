import type { GenerateLora, GenerateModelConfig } from '@workbench/generation/types';
import type { PromptHistoryItem } from '@workbench/types';
import type { ChangeEvent } from 'react';

import { Field } from '@workbench/components/ui';

import { PositivePromptActions } from './PositivePromptActions';
import { PROMPT_ATTENTION_TARGET_PROPS } from './promptAttentionHotkeys';
import { registerPositivePromptElement } from './promptFocus';
import { resetPromptHistoryNavigation } from './promptHistoryNavigation';
import { PromptTextarea } from './PromptTextarea';

interface PositivePromptFieldProps {
  heightPx: number;
  loras: GenerateLora[];
  selectedModel: GenerateModelConfig | undefined;
  showSyntaxHighlighting: boolean;
  value: string;
  onChange: (value: string) => void;
  onResizeEnd: (heightPx: number) => void;
  onUsePrompt: (prompt: PromptHistoryItem) => void;
}

export const PositivePromptField = ({
  heightPx,
  loras,
  onChange,
  onResizeEnd,
  onUsePrompt,
  selectedModel,
  showSyntaxHighlighting,
  value,
}: PositivePromptFieldProps) => (
  <Field
    label="Prompt"
    labelEnd={
      <PositivePromptActions
        loras={loras}
        positivePrompt={value}
        selectedModel={selectedModel}
        onPositivePromptChange={onChange}
        onUsePrompt={onUsePrompt}
      />
    }
  >
    <PromptTextarea
      {...PROMPT_ATTENTION_TARGET_PROPS}
      aria-label="Positive prompt"
      defaultHeightPx={heightPx}
      maxHeightPx={360}
      minHeightPx={96}
      resizeHandleAriaLabel="Resize positive prompt"
      size="xs"
      fontFamily="mono"
      showSyntaxHighlighting={showSyntaxHighlighting}
      textareaRef={registerPositivePromptElement}
      value={value}
      onChange={(event: ChangeEvent<HTMLTextAreaElement>) => {
        resetPromptHistoryNavigation();
        onChange(event.currentTarget.value);
      }}
      onResizeEnd={onResizeEnd}
    />
  </Field>
);
