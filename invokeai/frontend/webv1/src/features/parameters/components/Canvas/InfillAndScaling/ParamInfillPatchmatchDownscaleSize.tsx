import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectInfillMethod,
  selectInfillPatchmatchDownscaleSize,
  setInfillPatchmatchDownscaleSize,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 1,
  sliderMin: 1,
  sliderMax: 10,
  numberInputMin: 1,
  numberInputMax: 10,
  fineStep: 1,
  coarseStep: 1,
};

const ParamInfillPatchmatchDownscaleSize = () => {
  const dispatch = useAppDispatch();
  const infillMethod = useAppSelector(selectInfillMethod);
  const infillPatchmatchDownscaleSize = useAppSelector(selectInfillPatchmatchDownscaleSize);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setInfillPatchmatchDownscaleSize(v));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={infillMethod !== 'patchmatch'}>
      <InformationalPopover feature="patchmatchDownScaleSize">
        <FormLabel>{t('parameters.patchmatchDownScaleSize')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={infillPatchmatchDownscaleSize}
        onChange={handleChange}
        marks
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
      />
      <CompositeNumberInput
        value={infillPatchmatchDownscaleSize}
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

export default memo(ParamInfillPatchmatchDownscaleSize);
