import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectInfillMethod, selectInfillTileSize, setInfillTileSize } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 32,
  sliderMin: 16,
  sliderMax: 64,
  numberInputMin: 16,
  numberInputMax: 256,
  fineStep: 1,
  coarseStep: 1,
};

const ParamInfillTileSize = () => {
  const dispatch = useAppDispatch();
  const infillTileSize = useAppSelector(selectInfillTileSize);
  const infillMethod = useAppSelector(selectInfillMethod);

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
        value={infillTileSize}
        onChange={handleChange}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        marks
      />
      <CompositeNumberInput
        value={infillTileSize}
        onChange={handleChange}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
      />
    </FormControl>
  );
};

export default memo(ParamInfillTileSize);
