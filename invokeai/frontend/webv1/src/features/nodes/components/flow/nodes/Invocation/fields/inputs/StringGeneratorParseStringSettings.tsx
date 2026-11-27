import { Flex, FormControl, FormLabel, Input } from '@invoke-ai/ui-library';
import { GeneratorTextareaWithFileUpload } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/GeneratorTextareaWithFileUpload';
import type { StringGeneratorParseString } from 'features/nodes/types/field';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type StringGeneratorParseStringSettingsProps = {
  state: StringGeneratorParseString;
  onChange: (state: StringGeneratorParseString) => void;
};
export const StringGeneratorParseStringSettings = memo(
  ({ state, onChange }: StringGeneratorParseStringSettingsProps) => {
    const { t } = useTranslation();

    const onChangeSplitOn = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        onChange({ ...state, splitOn: e.target.value });
      },
      [onChange, state]
    );

    const onChangeInput = useCallback(
      (input: string) => {
        onChange({ ...state, input });
      },
      [onChange, state]
    );

    return (
      <Flex gap={2} flexDir="column">
        <FormControl orientation="vertical">
          <FormLabel>{t('nodes.splitOn')}</FormLabel>
          <Input value={state.splitOn} onChange={onChangeSplitOn} />
        </FormControl>
        <GeneratorTextareaWithFileUpload value={state.input} onChange={onChangeInput} />
      </Flex>
    );
  }
);
StringGeneratorParseStringSettings.displayName = 'StringGeneratorParseStringSettings';
