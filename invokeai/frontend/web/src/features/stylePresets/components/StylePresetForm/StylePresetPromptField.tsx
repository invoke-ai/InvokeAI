import { Button, Flex, FormControl, FormLabel, Textarea } from '@invoke-ai/ui-library';
import { PRESET_PLACEHOLDER } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import type { ChangeEventHandler } from 'react';
import { useCallback, useMemo, useRef } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { StylePresetFormData } from './StylePresetForm';

interface Props extends UseControllerProps<StylePresetFormData, 'negativePrompt' | 'positivePrompt'> {
  label: string;
}

export const StylePresetPromptField = (props: Props) => {
  const { field } = useController(props);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();

  const onChange = useCallback<ChangeEventHandler<HTMLTextAreaElement>>(
    (v) => {
      field.onChange(v.target.value);
    },
    [field]
  );

  const value = useMemo(() => {
    return field.value;
  }, [field.value]);

  const insertPromptPlaceholder = useCallback(() => {
    if (textareaRef.current) {
      const cursorPos = textareaRef.current.selectionStart;
      const textBeforeCursor = value.slice(0, cursorPos);
      const textAfterCursor = value.slice(cursorPos);
      const newValue = textBeforeCursor + PRESET_PLACEHOLDER + textAfterCursor;

      field.onChange(newValue);
    } else {
      field.onChange(value + PRESET_PLACEHOLDER);
    }

    textareaRef.current?.focus();
  }, [value, field, textareaRef]);

  const isPromptPresent = useMemo(() => value?.includes(PRESET_PLACEHOLDER), [value]);

  return (
    <FormControl orientation="vertical" gap={3}>
      <Flex alignItems="center" gap={2}>
        <FormLabel>{props.label}</FormLabel>
        <Button
          onClick={insertPromptPlaceholder}
          size="xs"
          aria-label={t('stylePresets.insertPlaceholder')}
          isDisabled={isPromptPresent}
        >
          {t('stylePresets.insertPlaceholder')}
        </Button>
      </Flex>

      <Textarea size="sm" ref={textareaRef} value={value} onChange={onChange} />
    </FormControl>
  );
};
