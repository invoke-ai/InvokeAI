import { CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { FloatGeneratorUniformRandomDistribution } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type FloatGeneratorUniformRandomDistributionSettingsProps = {
  state: FloatGeneratorUniformRandomDistribution;
  onChange: (state: FloatGeneratorUniformRandomDistribution) => void;
};
export const FloatGeneratorUniformRandomDistributionSettings = memo(
  ({ state, onChange }: FloatGeneratorUniformRandomDistributionSettingsProps) => {
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

    return (
      <Flex gap={2} alignItems="flex-end">
        <FormControl orientation="vertical">
          <FormLabel>{t('common.min')}</FormLabel>
          <CompositeNumberInput value={state.min} onChange={onChangeMin} min={-Infinity} max={Infinity} step={0.01} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.max')}</FormLabel>
          <CompositeNumberInput value={state.max} onChange={onChangeMax} min={-Infinity} max={Infinity} step={0.01} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.count')}</FormLabel>
          <CompositeNumberInput value={state.count} onChange={onChangeCount} min={1} max={Infinity} />
        </FormControl>
      </Flex>
    );
  }
);
FloatGeneratorUniformRandomDistributionSettings.displayName = 'FloatGeneratorUniformRandomDistributionSettings';
