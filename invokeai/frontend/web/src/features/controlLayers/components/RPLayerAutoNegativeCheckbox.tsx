import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isMaskedGuidanceLayer,
  maskLayerAutoNegativeChanged,
  selectRegionalPromptsSlice,
} from 'features/controlLayers/store/regionalPromptsSlice';
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
      createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isMaskedGuidanceLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return layer.autoNegative;
      }),
    [layerId]
  );
  const autoNegative = useAppSelector(selectAutoNegative);
  return autoNegative;
};

export const MaskedGuidanceLayerAutoNegativeCheckbox = memo(({ layerId }: Props) => {
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
      <FormLabel m={0}>{t('regionalPrompts.autoNegative')}</FormLabel>
      <Checkbox size="md" isChecked={autoNegative === 'invert'} onChange={onChange} />
    </FormControl>
  );
});

MaskedGuidanceLayerAutoNegativeCheckbox.displayName = 'MaskedGuidanceLayerAutoNegativeCheckbox';
