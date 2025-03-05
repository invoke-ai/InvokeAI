import { Checkbox, CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { GeneratorTextareaWithFileUpload } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/GeneratorTextareaWithFileUpload';
import type { StringGeneratorDynamicPromptsRandom } from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type StringGeneratorDynamicPromptsRandomSettingsProps = {
  state: StringGeneratorDynamicPromptsRandom;
  onChange: (state: StringGeneratorDynamicPromptsRandom) => void;
};
export const StringGeneratorDynamicPromptsRandomSettings = memo(
  ({ state, onChange }: StringGeneratorDynamicPromptsRandomSettingsProps) => {
    const { t } = useTranslation();

    const onChangeInput = useCallback(
      (input: string) => {
        onChange({ ...state, input });
      },
      [onChange, state]
    );
    const onChangeCount = useCallback(
      (v: number) => {
        onChange({ ...state, count: v });
      },
      [onChange, state]
    );
    const onToggleSeed = useCallback(() => {
      onChange({ ...state, seed: isNil(state.seed) ? 0 : null });
    }, [onChange, state]);
    const onChangeSeed = useCallback(
      (seed?: number | null) => {
        onChange({ ...state, seed });
      },
      [onChange, state]
    );

    return (
      <Flex gap={2} flexDir="column">
        <Flex gap={2}>
          <FormControl orientation="vertical">
            <FormLabel alignItems="center" justifyContent="space-between" display="flex" w="full" pe={0.5}>
              {t('common.seed')}
              <Checkbox onChange={onToggleSeed} isChecked={!isNil(state.seed)} />
            </FormLabel>
            <CompositeNumberInput
              isDisabled={isNil(state.seed)}
              // This cast is save only because we disable the element when seed is not a number - the `...` is
              // rendered in the input field in this case
              value={state.seed ?? ('...' as unknown as number)}
              onChange={onChangeSeed}
              min={-Infinity}
              max={Infinity}
            />
          </FormControl>
          <FormControl orientation="vertical">
            <FormLabel>{t('common.count')}</FormLabel>
            <CompositeNumberInput value={state.count} onChange={onChangeCount} min={1} max={1000} />
          </FormControl>
        </Flex>
        <GeneratorTextareaWithFileUpload value={state.input} onChange={onChangeInput} />
      </Flex>
    );
  }
);
StringGeneratorDynamicPromptsRandomSettings.displayName = 'StringGeneratorDynamicPromptsRandomSettings';
