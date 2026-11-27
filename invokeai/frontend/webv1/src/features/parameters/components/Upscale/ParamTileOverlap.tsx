import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectTileOverlap, tileOverlapChanged } from 'features/parameters/store/upscaleSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const initial = 128;
const sliderMin = 32;
const sliderMax = 256;
const numberInputMin = 16;
const numberInputMax = 512;
const coarseStep = 16;
const fineStep = 8;
const marks = [sliderMin, 128, sliderMax];

const ParamTileOverlap = () => {
  const tileOverlap = useAppSelector(selectTileOverlap);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback(
    (v: number) => {
      dispatch(tileOverlapChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('upscaling.tileOverlap')}</FormLabel>
      <CompositeSlider
        value={tileOverlap}
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={tileOverlap}
        defaultValue={initial}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default ParamTileOverlap;
