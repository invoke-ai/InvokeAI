import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { caLayerWeightChanged, selectCALayer } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  layerId: string;
};

const formatValue = (v: number) => v.toFixed(2);
const marks = [0, 1, 2];

export const CALayerWeight = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const weight = useAppSelector((s) => selectCALayer(s.controlLayers.present, layerId).controlAdapter.weight);
  const initial = useAppSelector((s) => s.config.sd.ca.weight.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.ca.weight.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.ca.weight.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.ca.weight.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.ca.weight.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.ca.weight.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.ca.weight.fineStep);

  const onChange = useCallback(
    (weight: number) => {
      dispatch(caLayerWeightChanged({ layerId, weight }));
    },
    [dispatch, layerId]
  );

  return (
    <FormControl orientation="horizontal">
      <InformationalPopover feature="controlNetWeight">
        <FormLabel m={0}>{t('controlnet.weight')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={weight}
        onChange={onChange}
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        marks={marks}
        formatValue={formatValue}
      />
      <CompositeNumberInput
        value={weight}
        onChange={onChange}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        maxW={20}
        defaultValue={initial}
      />
    </FormControl>
  );
});

CALayerWeight.displayName = 'CALayerWeight';
