import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setRefinerPositiveAestheticScore } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ sdxl, hotkeys }) => {
    const { refinerPositiveAestheticScore } = sdxl;
    const { shift } = hotkeys;

    return {
      refinerPositiveAestheticScore,
      shift,
    };
  }
);

const ParamSDXLRefinerPositiveAestheticScore = () => {
  const { refinerPositiveAestheticScore, shift } = useAppSelector(selector);

  const isRefinerAvailable = useIsRefinerAvailable();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setRefinerPositiveAestheticScore(v)),
    [dispatch]
  );

  const handleReset = useCallback(
    () => dispatch(setRefinerPositiveAestheticScore(6)),
    [dispatch]
  );

  return (
    <IAISlider
      label={t('sdxl.posAestheticScore')}
      step={shift ? 0.1 : 0.5}
      min={1}
      max={10}
      onChange={handleChange}
      handleReset={handleReset}
      value={refinerPositiveAestheticScore}
      sliderNumberInputProps={{ max: 10 }}
      withInput
      withReset
      withSliderMarks
      isInteger={false}
      isDisabled={!isRefinerAvailable}
    />
  );
};

export default memo(ParamSDXLRefinerPositiveAestheticScore);
