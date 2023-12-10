import { Tooltip } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setHrfStrength } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ generation, hotkeys, config }) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      config.sd.hrfStrength;
    const { hrfStrength, hrfEnabled } = generation;
    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      hrfStrength,
      initial,
      min,
      sliderMax,
      inputMax,
      step,
      hrfEnabled,
    };
  }
);

const ParamHrfStrength = () => {
  const { hrfStrength, initial, min, sliderMax, step, hrfEnabled } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleHrfStrengthReset = useCallback(() => {
    dispatch(setHrfStrength(initial));
  }, [dispatch, initial]);

  const handleHrfStrengthChange = useCallback(
    (v: number) => {
      dispatch(setHrfStrength(v));
    },
    [dispatch]
  );

  return (
    <Tooltip label={t('hrf.strengthTooltip')} placement="right" hasArrow>
      <IAISlider
        label={t('parameters.denoisingStrength')}
        min={min}
        max={sliderMax}
        step={step}
        value={hrfStrength}
        onChange={handleHrfStrengthChange}
        withSliderMarks
        withInput
        withReset
        handleReset={handleHrfStrengthReset}
        isDisabled={!hrfEnabled}
      />
    </Tooltip>
  );
};

export default memo(ParamHrfStrength);
