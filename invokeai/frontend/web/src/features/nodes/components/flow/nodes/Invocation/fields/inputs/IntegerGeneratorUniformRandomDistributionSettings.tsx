import { CompositeNumberInput, Flex, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import type { IntegerGeneratorUniformRandomDistribution } from 'features/nodes/types/field';
import { getIntegerGeneratorUniformRandomDistributionDefaults } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type IntegerGeneratorUniformRandomDistributionSettingsProps = {
  state: IntegerGeneratorUniformRandomDistribution;
  onChange: (state: IntegerGeneratorUniformRandomDistribution) => void;
};
export const IntegerGeneratorUniformRandomDistributionSettings = memo(
  ({ state, onChange }: IntegerGeneratorUniformRandomDistributionSettingsProps) => {
    const { t } = useTranslation();

    const onChangeMin = useCallback(
      (min: number) => {
        onChange({ ...state, min });
      },
      [onChange, state]
    );
    const onChangeMax = useCallback(
      (max: number) => {
        onChange({ ...state, max });
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
      onChange(getIntegerGeneratorUniformRandomDistributionDefaults());
    }, [onChange]);

    return (
      <Flex gap={2} alignItems="flex-end">
        <FormControl orientation="vertical">
          <FormLabel>{t('common.min')}</FormLabel>
          <CompositeNumberInput value={state.min} onChange={onChangeMin} min={-Infinity} max={Infinity} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.max')}</FormLabel>
          <CompositeNumberInput value={state.max} onChange={onChangeMax} min={-Infinity} max={Infinity} />
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
IntegerGeneratorUniformRandomDistributionSettings.displayName = 'IntegerGeneratorUniformRandomDistributionSettings';
