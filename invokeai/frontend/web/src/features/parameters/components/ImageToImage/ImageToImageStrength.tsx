import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setImg2imgStrength } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ generation, config }) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      config.sd.img2imgStrength;
    const { img2imgStrength } = generation;

    return {
      img2imgStrength,
      initial,
      min,
      sliderMax,
      inputMax,
      step: coarseStep,
      fineStep,
    };
  }
);

const marks = [0, 0.5, 1];

const ImageToImageStrength = () => {
  const { img2imgStrength, initial, min, sliderMax, inputMax, step, fineStep } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setImg2imgStrength(v)),
    [dispatch]
  );

  return (
    <InvControl
      label={`${t('parameters.denoisingStrength')}`}
      feature="paramDenoisingStrength"
    >
      <InvSlider
        step={step}
        fineStep={fineStep}
        min={min}
        max={sliderMax}
        onChange={handleChange}
        value={img2imgStrength}
        defaultValue={initial}
        marks={marks}
        withNumberInput
        numberInputMax={inputMax}
      />
    </InvControl>
  );
};

export default memo(ImageToImageStrength);
