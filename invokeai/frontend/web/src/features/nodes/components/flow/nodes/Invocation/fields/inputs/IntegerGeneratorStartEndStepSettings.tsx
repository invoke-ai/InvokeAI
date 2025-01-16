import { CompositeNumberInput, Flex, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import { getIntegerGeneratorStartEndStepDefaults, type IntegerGeneratorStartEndStep } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type IntegerGeneratorStartEndStepSettingsProps = {
  state: IntegerGeneratorStartEndStep;
  onChange: (state: IntegerGeneratorStartEndStep) => void;
};
export const IntegerGeneratorStartEndStepSettings = memo(
  ({ state, onChange }: IntegerGeneratorStartEndStepSettingsProps) => {
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
    const onChangeEnd = useCallback(
      (end: number) => {
        onChange({ ...state, end });
      },
      [onChange, state]
    );
    const onReset = useCallback(() => {
      onChange(getIntegerGeneratorStartEndStepDefaults());
    }, [onChange]);

    return (
      <Flex gap={2} alignItems="flex-end">
        <FormControl orientation="vertical">
          <FormLabel>{t('common.start')}</FormLabel>
          <CompositeNumberInput value={state.start} onChange={onChangeStart} min={-Infinity} max={Infinity} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.end')}</FormLabel>
          <CompositeNumberInput value={state.end} onChange={onChangeEnd} min={-Infinity} max={Infinity} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.step')}</FormLabel>
          <CompositeNumberInput value={state.step} onChange={onChangeStep} min={-Infinity} max={Infinity} />
        </FormControl>
        <IconButton aria-label="Reset" icon={<PiArrowCounterClockwiseBold />} onClick={onReset} variant="ghost" />
      </Flex>
    );
  }
);
IntegerGeneratorStartEndStepSettings.displayName = 'IntegerGeneratorStartEndStepSettings';
