import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isRegionalGuidanceLayer,
  maskLayerAutoNegativeChanged,
  selectControlLayersSlice,
} from 'features/controlLayers/store/controlLayersSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

const useAutoNegative = (layerId: string) => {
  const selectAutoNegative = useMemo(
    () =>
      createSelector(selectControlLayersSlice, (controlLayers) => {
        const layer = controlLayers.present.layers.find((l) => l.id === layerId);
        assert(isRegionalGuidanceLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return layer.autoNegative;
      }),
    [layerId]
  );
  const autoNegative = useAppSelector(selectAutoNegative);
  return autoNegative;
};

export const RGLayerAutoNegativeCheckbox = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoNegative = useAutoNegative(layerId);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(maskLayerAutoNegativeChanged({ layerId, autoNegative: e.target.checked ? 'invert' : 'off' }));
    },
    [dispatch, layerId]
  );

  return (
    <FormControl gap={2}>
      <FormLabel m={0}>{t('controlLayers.autoNegative')}</FormLabel>
      <Checkbox size="md" isChecked={autoNegative === 'invert'} onChange={onChange} />
    </FormControl>
  );
});

RGLayerAutoNegativeCheckbox.displayName = 'RGLayerAutoNegativeCheckbox';
