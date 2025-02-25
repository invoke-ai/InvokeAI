import { CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { GeneratorTextareaWithFileUpload } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/GeneratorTextareaWithFileUpload';
import type { StringGeneratorDynamicPromptsCombinatorial } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type StringGeneratorDynamicPromptsCombinatorialSettingsProps = {
  state: StringGeneratorDynamicPromptsCombinatorial;
  onChange: (state: StringGeneratorDynamicPromptsCombinatorial) => void;
};
export const StringGeneratorDynamicPromptsCombinatorialSettings = memo(
  ({ state, onChange }: StringGeneratorDynamicPromptsCombinatorialSettingsProps) => {
    const { t } = useTranslation();

    const onChangeInput = useCallback(
      (input: string) => {
        onChange({ ...state, input });
      },
      [onChange, state]
    );
    const onChangeMaxPrompts = useCallback(
      (v: number) => {
        onChange({ ...state, maxPrompts: v });
      },
      [onChange, state]
    );

    return (
      <Flex gap={2} flexDir="column">
        <FormControl orientation="vertical">
          <FormLabel>{t('dynamicPrompts.maxPrompts')}</FormLabel>
          <CompositeNumberInput value={state.maxPrompts} onChange={onChangeMaxPrompts} min={1} max={1000} w="full" />
        </FormControl>
        <GeneratorTextareaWithFileUpload value={state.input} onChange={onChangeInput} />
      </Flex>
    );
  }
);
StringGeneratorDynamicPromptsCombinatorialSettings.displayName = 'StringGeneratorDynamicPromptsCombinatorialSettings';
