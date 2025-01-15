import { CompositeNumberInput, Flex, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import {
  type FloatRangeStartStepCountGenerator,
  getDefaultFloatRangeStartStepCountGenerator,
} from 'features/nodes/types/generators';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type FloatRangeGeneratorProps = {
  state: FloatRangeStartStepCountGenerator;
  onChange: (state: FloatRangeStartStepCountGenerator) => void;
};

export const FloatRangeGenerator = memo(({ state, onChange }: FloatRangeGeneratorProps) => {
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
    onChange(getDefaultFloatRangeStartStepCountGenerator());
  }, [onChange]);

  return (
    <Flex gap={1} alignItems="flex-end" p={1}>
      <FormControl orientation="vertical" gap={1}>
        <FormLabel m={0}>{t('common.start')}</FormLabel>
        <CompositeNumberInput value={state.start} onChange={onChangeStart} min={-Infinity} max={Infinity} step={0.01} />
      </FormControl>
      <FormControl orientation="vertical" gap={1}>
        <FormLabel m={0}>{t('common.count')}</FormLabel>
        <CompositeNumberInput value={state.count} onChange={onChangeCount} min={1} max={Infinity} />
      </FormControl>
      <FormControl orientation="vertical" gap={1}>
        <FormLabel m={0}>{t('common.step')}</FormLabel>
        <CompositeNumberInput value={state.step} onChange={onChangeStep} min={-Infinity} max={Infinity} step={0.01} />
      </FormControl>
      <IconButton
        onClick={onReset}
        aria-label={t('common.reset')}
        icon={<PiArrowCounterClockwiseBold />}
        variant="ghost"
      />
    </Flex>
  );
});

FloatRangeGenerator.displayName = 'FloatRangeGenerator';
