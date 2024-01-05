import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import {
  selectGenerationSlice,
  setCfgScale,
} from 'features/parameters/store/generationSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  selectGenerationSlice,
  selectConfigSlice,
  (generation, config) => {
    const { min, inputMax, sliderMax, coarseStep, fineStep, initial } =
      config.sd.guidance;
    const { cfgScale } = generation;

    return {
      marks: [min, Math.floor(sliderMax / 2), sliderMax],
      cfgScale,
      min,
      inputMax,
      sliderMax,
      coarseStep,
      fineStep,
      initial,
    };
  }
);

const ParamCFGScale = () => {
  const {
    cfgScale,
    min,
    inputMax,
    sliderMax,
    coarseStep,
    fineStep,
    initial,
    marks,
  } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => dispatch(setCfgScale(v)),
    [dispatch]
  );

  return (
    <InvControl label={t('parameters.cfgScale')} feature="paramCFGScale">
      <InvSlider
        value={cfgScale}
        defaultValue={initial}
        min={min}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        withNumberInput
        marks={marks}
        numberInputMax={inputMax}
      />
    </InvControl>
  );
};

export default memo(ParamCFGScale);
