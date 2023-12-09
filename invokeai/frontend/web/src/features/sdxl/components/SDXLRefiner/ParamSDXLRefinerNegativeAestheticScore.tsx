import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setRefinerNegativeAestheticScore } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ sdxl, hotkeys }) => {
    const { refinerNegativeAestheticScore } = sdxl;
    const { shift } = hotkeys;

    return {
      refinerNegativeAestheticScore,
      shift,
    };
  }
);

const ParamSDXLRefinerNegativeAestheticScore = () => {
  const { refinerNegativeAestheticScore, shift } = useAppSelector(selector);

  const isRefinerAvailable = useIsRefinerAvailable();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setRefinerNegativeAestheticScore(v)),
    [dispatch]
  );

  const handleReset = useCallback(
    () => dispatch(setRefinerNegativeAestheticScore(2.5)),
    [dispatch]
  );

  return (
    <IAISlider
      label={t('sdxl.negAestheticScore')}
      step={shift ? 0.1 : 0.5}
      min={1}
      max={10}
      onChange={handleChange}
      handleReset={handleReset}
      value={refinerNegativeAestheticScore}
      sliderNumberInputProps={{ max: 10 }}
      withInput
      withReset
      withSliderMarks
      isInteger={false}
      isDisabled={!isRefinerAvailable}
    />
  );
};

export default memo(ParamSDXLRefinerNegativeAestheticScore);
