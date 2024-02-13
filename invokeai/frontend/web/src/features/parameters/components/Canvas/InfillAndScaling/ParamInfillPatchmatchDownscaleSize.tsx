import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setInfillPatchmatchDownscaleSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamInfillPatchmatchDownscaleSize = () => {
  const dispatch = useAppDispatch();
  const infillMethod = useAppSelector((s) => s.generation.infillMethod);
  const infillPatchmatchDownscaleSize = useAppSelector((s) => s.generation.infillPatchmatchDownscaleSize);
  const initial = useAppSelector((s) => s.config.sd.infillPatchmatchDownscaleSize.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.infillPatchmatchDownscaleSize.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.infillPatchmatchDownscaleSize.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.infillPatchmatchDownscaleSize.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.infillPatchmatchDownscaleSize.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.infillPatchmatchDownscaleSize.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.infillPatchmatchDownscaleSize.fineStep);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setInfillPatchmatchDownscaleSize(v));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={infillMethod !== 'patchmatch'}>
      <FormLabel>{t('parameters.patchmatchDownScaleSize')}</FormLabel>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={infillPatchmatchDownscaleSize}
        defaultValue={initial}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        value={infillPatchmatchDownscaleSize}
        defaultValue={initial}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamInfillPatchmatchDownscaleSize);
