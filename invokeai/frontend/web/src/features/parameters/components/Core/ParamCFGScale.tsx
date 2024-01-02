import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setCfgScale } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ generation, config }) => {
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

  const onReset = useCallback(() => {
    dispatch(setCfgScale(initial));
  }, [dispatch, initial]);

  return (
    <InvControl label={t('parameters.cfgScale')} feature="paramCFGScale">
      <InvSlider
        value={cfgScale}
        min={min}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        onReset={onReset}
        withNumberInput
        marks={marks}
        numberInputMax={inputMax}
      />
    </InvControl>
  );
};

export default memo(ParamCFGScale);
