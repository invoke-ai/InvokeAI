import { CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { FloatGeneratorArithmeticSequence } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type FloatGeneratorArithmeticSequenceSettingsProps = {
  state: FloatGeneratorArithmeticSequence;
  onChange: (state: FloatGeneratorArithmeticSequence) => void;
};
export const FloatGeneratorArithmeticSequenceSettings = memo(
  ({ state, onChange }: FloatGeneratorArithmeticSequenceSettingsProps) => {
    const { t } = useTranslation();

    const onChangeStart = useCallback(
      (start: number) => {
        onChange({ ...state, start });
      },
      [onChange, state]
    );
    const onChangeStep = useCallback(
      (step: number) => {
        onChange({ ...state, step });
      },
      [onChange, state]
    );
    const onChangeCount = useCallback(
      (count: number) => {
        onChange({ ...state, count });
      },
      [onChange, state]
    );

    return (
      <Flex gap={2} alignItems="flex-end">
        <FormControl orientation="vertical">
          <FormLabel>{t('common.start')}</FormLabel>
          <CompositeNumberInput
            value={state.start}
            onChange={onChangeStart}
            min={-Infinity}
            max={Infinity}
            step={0.01}
          />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.step')}</FormLabel>
          <CompositeNumberInput value={state.step} onChange={onChangeStep} min={-Infinity} max={Infinity} step={0.01} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.count')}</FormLabel>
          <CompositeNumberInput value={state.count} onChange={onChangeCount} min={1} max={Infinity} />
        </FormControl>
      </Flex>
    );
  }
);
FloatGeneratorArithmeticSequenceSettings.displayName = 'FloatGeneratorArithmeticSequenceSettings';
