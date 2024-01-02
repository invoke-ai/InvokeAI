import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setRefinerCFGScale } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector([stateSelector], ({ config }) => {
  const { min, inputMax, sliderMax, coarseStep, fineStep, initial } =
    config.sd.guidance;

  return {
    marks: [min, Math.floor(sliderMax / 2), sliderMax],
    min,
    inputMax,
    sliderMax,
    coarseStep,
    fineStep,
    initial,
  };
});

const ParamSDXLRefinerCFGScale = () => {
  const refinerCFGScale = useAppSelector((state) => state.sdxl.refinerCFGScale);
  const { marks, min, inputMax, sliderMax, coarseStep, fineStep, initial } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => dispatch(setRefinerCFGScale(v)),
    [dispatch]
  );

  return (
    <InvControl label={t('sdxl.cfgScale')}>
      <InvSlider
        value={refinerCFGScale}
        defaultValue={initial}
        min={min}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        withNumberInput
        numberInputMax={inputMax}
        marks={marks}
      />
    </InvControl>
  );
};

export default memo(ParamSDXLRefinerCFGScale);
