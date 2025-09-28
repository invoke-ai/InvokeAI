import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectInfillMethod,
  selectInfillPatchmatchDownscaleSize,
  setInfillPatchmatchDownscaleSize,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import { selectInfillPatchmatchDownscaleSizeConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamInfillPatchmatchDownscaleSize = () => {
  const dispatchParams = useParamsDispatch();
  const infillMethod = useAppSelector(selectInfillMethod);
  const infillPatchmatchDownscaleSize = useAppSelector(selectInfillPatchmatchDownscaleSize);
  const config = useAppSelector(selectInfillPatchmatchDownscaleSizeConfig);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatchParams(setInfillPatchmatchDownscaleSize, v);
    },
    [dispatchParams]
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
