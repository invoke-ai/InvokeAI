import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectInfillMethod,
  selectInfillPatchmatchDownscaleSize,
  setInfillPatchmatchDownscaleSize,
} from 'features/controlLayers/store/paramsSlice';
import { selectInfillPatchmatchDownscaleSizeConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamInfillPatchmatchDownscaleSize = () => {
  const dispatch = useAppDispatch();
  const infillMethod = useAppSelector(selectInfillMethod);
  const infillPatchmatchDownscaleSize = useAppSelector(selectInfillPatchmatchDownscaleSize);
  const config = useAppSelector(selectInfillPatchmatchDownscaleSizeConfig);

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
        defaultValue={config.initial}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
      />
      <CompositeNumberInput
        value={infillPatchmatchDownscaleSize}
        onChange={handleChange}
        defaultValue={config.initial}
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
      />
    </FormControl>
  );
};

export default memo(ParamInfillPatchmatchDownscaleSize);
