import { CompositeNumberInput, Flex, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import type { IntegerGeneratorArithmeticSequence } from 'features/nodes/types/field';
import { getIntegerGeneratorArithmeticSequenceDefaults } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type IntegerGeneratorArithmeticSequenceSettingsProps = {
  state: IntegerGeneratorArithmeticSequence;
  onChange: (state: IntegerGeneratorArithmeticSequence) => void;
};
export const IntegerGeneratorArithmeticSequenceSettings = memo(
  ({ state, onChange }: IntegerGeneratorArithmeticSequenceSettingsProps) => {
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
    const onReset = useCallback(() => {
      onChange(getIntegerGeneratorArithmeticSequenceDefaults());
    }, [onChange]);

    return (
      <Flex gap={2} alignItems="flex-end">
        <FormControl orientation="vertical">
          <FormLabel>{t('common.start')}</FormLabel>
          <CompositeNumberInput value={state.start} onChange={onChangeStart} min={-Infinity} max={Infinity} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.step')}</FormLabel>
          <CompositeNumberInput value={state.step} onChange={onChangeStep} min={-Infinity} max={Infinity} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.count')}</FormLabel>
          <CompositeNumberInput value={state.count} onChange={onChangeCount} min={1} max={Infinity} />
        </FormControl>
        <IconButton aria-label="Reset" icon={<PiArrowCounterClockwiseBold />} onClick={onReset} variant="ghost" />
      </Flex>
    );
  }
);
IntegerGeneratorArithmeticSequenceSettings.displayName = 'IntegerGeneratorArithmeticSequenceSettings';
