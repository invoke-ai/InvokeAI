import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  globalMaskLayerOpacityChanged,
  initialControlLayersState,
} from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [0, 25, 50, 75, 100];
const formatPct = (v: number | string) => `${v} %`;

export const GlobalMaskLayerOpacity = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const globalMaskLayerOpacity = useAppSelector((s) =>
    Math.round(s.controlLayers.present.globalMaskLayerOpacity * 100)
  );
  const onChange = useCallback(
    (v: number) => {
      dispatch(globalMaskLayerOpacityChanged(v / 100));
    },
    [dispatch]
  );
  return (
    <FormControl orientation="vertical">
      <FormLabel m={0}>{t('controlLayers.globalMaskOpacity')}</FormLabel>
      <Flex gap={4}>
        <CompositeSlider
          min={0}
          max={100}
          step={1}
          value={globalMaskLayerOpacity}
          defaultValue={initialControlLayersState.globalMaskLayerOpacity * 100}
          onChange={onChange}
          marks={marks}
          minW={48}
        />
        <CompositeNumberInput
          min={0}
          max={100}
          step={1}
          value={globalMaskLayerOpacity}
          defaultValue={initialControlLayersState.globalMaskLayerOpacity * 100}
          onChange={onChange}
          w={28}
          format={formatPct}
        />
      </Flex>
    </FormControl>
  );
});

GlobalMaskLayerOpacity.displayName = 'GlobalMaskLayerOpacity';
