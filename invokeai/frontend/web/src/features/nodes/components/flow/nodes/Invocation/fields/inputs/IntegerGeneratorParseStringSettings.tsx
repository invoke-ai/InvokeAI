import { Flex, FormControl, FormLabel, Input, Textarea } from '@invoke-ai/ui-library';
import type { IntegerGeneratorParseString } from 'features/nodes/types/field';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type IntegerGeneratorParseStringSettingsProps = {
  state: IntegerGeneratorParseString;
  onChange: (state: IntegerGeneratorParseString) => void;
};
export const IntegerGeneratorParseStringSettings = memo(
  ({ state, onChange }: IntegerGeneratorParseStringSettingsProps) => {
    const { t } = useTranslation();

    const onChangeSplitOn = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        onChange({ ...state, splitOn: e.target.value });
      },
      [onChange, state]
    );

    const onChangeInput = useCallback(
      (e: ChangeEvent<HTMLTextAreaElement>) => {
        onChange({ ...state, input: e.target.value });
      },
      [onChange, state]
    );

    return (
      <Flex gap={2} flexDir="column">
        <FormControl orientation="vertical">
          <FormLabel>{t('nodes.splitOn')}</FormLabel>
          <Input value={state.splitOn} onChange={onChangeSplitOn} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.input')}</FormLabel>
          <Textarea
            className="nowheel nodrag nopan"
            value={state.input}
            onChange={onChangeInput}
            p={2}
            resize="none"
            rows={5}
          />
        </FormControl>
      </Flex>
    );
  }
);
IntegerGeneratorParseStringSettings.displayName = 'IntegerGeneratorParseStringSettings';
