import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectTileSize, tileSizeChanged } from 'features/parameters/store/upscaleSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const initial = 1024;
const sliderMin = 512;
const sliderMax = 1536;
const numberInputMin = 512;
const numberInputMax = 1536;
const coarseStep = 64;
const fineStep = 32;
const marks = [sliderMin, 1024, sliderMax];

const ParamTileSize = () => {
  const tileSize = useAppSelector(selectTileSize);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback(
    (v: number) => {
      dispatch(tileSizeChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('upscaling.tileSize')}</FormLabel>
      <CompositeSlider
        value={tileSize}
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={tileSize}
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

export default ParamTileSize;
