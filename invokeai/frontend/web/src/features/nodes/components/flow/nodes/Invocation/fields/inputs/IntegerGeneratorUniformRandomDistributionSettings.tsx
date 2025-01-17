import { Checkbox, CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { IntegerGeneratorUniformRandomDistribution } from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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
          <FormControl orientation="vertical">
            <FormLabel alignItems="center" justifyContent="space-between" m={0} display="flex" w="full">
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
        </Flex>
      </Flex>
    );
  }
);
IntegerGeneratorUniformRandomDistributionSettings.displayName = 'IntegerGeneratorUniformRandomDistributionSettings';
