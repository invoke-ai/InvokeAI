import { CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { FloatGeneratorLinearDistribution } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type FloatGeneratorLinearDistributionSettingsProps = {
  state: FloatGeneratorLinearDistribution;
  onChange: (state: FloatGeneratorLinearDistribution) => void;
};
export const FloatGeneratorLinearDistributionSettings = memo(
  ({ state, onChange }: FloatGeneratorLinearDistributionSettingsProps) => {
    const { t } = useTranslation();

    const onChangeStart = useCallback(
      (start: number) => {
        onChange({ ...state, start });
      },
      [onChange, state]
    );
    const onChangeEnd = useCallback(
      (end: number) => {
        onChange({ ...state, end });
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
          <FormLabel>{t('common.end')}</FormLabel>
          <CompositeNumberInput value={state.end} onChange={onChangeEnd} min={-Infinity} max={Infinity} step={0.01} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.count')}</FormLabel>
          <CompositeNumberInput value={state.count} onChange={onChangeCount} min={1} max={Infinity} />
        </FormControl>
      </Flex>
    );
  }
);
FloatGeneratorLinearDistributionSettings.displayName = 'FloatGeneratorLinearDistributionSettings';
