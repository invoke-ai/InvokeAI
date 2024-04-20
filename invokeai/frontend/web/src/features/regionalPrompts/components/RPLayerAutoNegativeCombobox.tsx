import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isParameterAutoNegative } from 'features/parameters/types/parameterSchemas';
import {
  isVectorMaskLayer,
  maskLayerAutoNegativeChanged,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

const options: ComboboxOption[] = [
  { label: 'Off', value: 'off' },
  { label: 'Invert', value: 'invert' },
];

type Props = {
  layerId: string;
};

const useAutoNegative = (layerId: string) => {
  const selectAutoNegative = useMemo(
    () =>
      createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isVectorMaskLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return layer.autoNegative;
      }),
    [layerId]
  );
  const autoNegative = useAppSelector(selectAutoNegative);
  return autoNegative;
};

export const RPLayerAutoNegativeCombobox = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const autoNegative = useAutoNegative(layerId);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterAutoNegative(v?.value)) {
        return;
      }
      dispatch(maskLayerAutoNegativeChanged({ layerId, autoNegative: v.value }));
    },
    [dispatch, layerId]
  );

  const value = useMemo(() => options.find((o) => o.value === autoNegative), [autoNegative]);

  return (
    <FormControl flexGrow={0} gap={2} w="min-content">
      <FormLabel m={0}>{t('regionalPrompts.autoNegative')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} isSearchable={false} sx={{ w: '5.2rem' }} />
    </FormControl>
  );
});

RPLayerAutoNegativeCombobox.displayName = 'RPLayerAutoNegativeCombobox';
