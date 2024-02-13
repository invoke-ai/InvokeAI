import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setInfillTileSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamInfillTileSize = () => {
  const dispatch = useAppDispatch();
  const infillTileSize = useAppSelector((s) => s.generation.infillTileSize);
  const initial = useAppSelector((s) => s.config.sd.infillTileSize.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.infillTileSize.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.infillTileSize.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.infillTileSize.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.infillTileSize.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.infillTileSize.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.infillTileSize.fineStep);

  const infillMethod = useAppSelector((s) => s.generation.infillMethod);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setInfillTileSize(v));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={infillMethod !== 'tile'}>
      <FormLabel>{t('parameters.tileSize')}</FormLabel>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        value={infillTileSize}
        defaultValue={initial}
        onChange={handleChange}
        step={coarseStep}
        fineStep={fineStep}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        value={infillTileSize}
        defaultValue={initial}
        onChange={handleChange}
        step={coarseStep}
        fineStep={fineStep}
      />
    </FormControl>
  );
};

export default memo(ParamInfillTileSize);
