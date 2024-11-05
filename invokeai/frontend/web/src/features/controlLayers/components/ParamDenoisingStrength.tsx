import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectImg2imgStrength, setImg2imgStrength } from 'features/controlLayers/store/paramsSlice';
import { selectImg2imgStrengthConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { selectCanvasSlice } from '../store/selectors';
import WavyLine from '../../../common/components/WavyLine';

const marks = [0, 0.5, 1];

export const ParamDenoisingStrength = memo(() => {
  const img2imgStrength = useAppSelector(selectImg2imgStrength);
  const dispatch = useAppDispatch();
  const isEnabled = useAppSelector(
    (s) => selectCanvasSlice(s).rasterLayers.entities.length > 0 && selectCanvasSlice(s).rasterLayers.entities.some(layer => layer.isEnabled)
  );


  const onChange = useCallback(
    (v: number) => {
      dispatch(setImg2imgStrength(v));
    },
    [dispatch]
  );

  const config = useAppSelector(selectImg2imgStrengthConfig);
  const { t } = useTranslation();

  return (
    <FormControl isDisabled={!isEnabled} py={2}>
      <InformationalPopover feature="paramDenoisingStrength">
        <FormLabel mr={0}>{`${t('parameters.denoisingStrength')}`}</FormLabel>
      </InformationalPopover>
      <WavyLine waviness={img2imgStrength * 10} isDisabled={!isEnabled}/>
      <CompositeSlider
        step={config.coarseStep}
        fineStep={config.fineStep}
        min={config.sliderMin}
        max={config.sliderMax}
        defaultValue={config.initial}
        onChange={onChange}
        value={img2imgStrength}
        marks={marks}
      />
      <CompositeNumberInput
        step={config.coarseStep}
        fineStep={config.fineStep}
        min={config.numberInputMin}
        max={config.numberInputMax}
        defaultValue={config.initial}
        onChange={onChange}
        value={img2imgStrength}
      />
    </FormControl>
  );
});

ParamDenoisingStrength.displayName = 'ParamDenoisingStrength';
