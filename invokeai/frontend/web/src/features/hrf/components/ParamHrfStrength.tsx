import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setHrfStrength } from 'features/hrf/store/hrfSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector([stateSelector], ({ hrf, config }) => {
  const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
    config.sd.hrfStrength;
  const { hrfStrength } = hrf;

  return {
    hrfStrength,
    initial,
    min,
    sliderMax,
    inputMax,
    step: coarseStep,
    fineStep,
  };
});

const ParamHrfStrength = () => {
  const { hrfStrength, initial, min, sliderMax, step, fineStep } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleHrfStrengthChange = useCallback(
    (v: number) => {
      dispatch(setHrfStrength(v));
    },
    [dispatch]
  );

  return (
    <InvControl label={t('parameters.denoisingStrength')}>
      <InvSlider
        min={min}
        max={sliderMax}
        step={step}
        fineStep={fineStep}
        value={hrfStrength}
        defaultValue={initial}
        onChange={handleHrfStrengthChange}
        marks
        withNumberInput
      />
    </InvControl>
  );
};

export default memo(ParamHrfStrength);
