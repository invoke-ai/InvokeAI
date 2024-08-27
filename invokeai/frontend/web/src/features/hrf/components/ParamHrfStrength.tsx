import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectHrfStrength, setHrfStrength } from 'features/hrf/store/hrfSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectHrfStrengthConfig = createSelector(selectConfigSlice, (config) => config.sd.hrfStrength);

const ParamHrfStrength = () => {
  const hrfStrength = useAppSelector(selectHrfStrength);
  const config = useAppSelector(selectHrfStrengthConfig);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfStrength(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramDenoisingStrength">
        <FormLabel>{`${t('parameters.denoisingStrength')}`}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        value={hrfStrength}
        defaultValue={config.initial}
        onChange={onChange}
        marks
      />
      <CompositeNumberInput
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        value={hrfStrength}
        defaultValue={config.initial}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamHrfStrength);
