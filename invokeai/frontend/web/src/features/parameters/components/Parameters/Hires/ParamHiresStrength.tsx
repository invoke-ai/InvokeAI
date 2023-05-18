import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import { setHiresStrength } from 'features/parameters/store/postprocessingSlice';
import { isEqual } from 'lodash-es';
import { useTranslation } from 'react-i18next';

const hiresStrengthSelector = createSelector(
  [postprocessingSelector],
  ({ hiresFix, hiresStrength }) => ({ hiresFix, hiresStrength }),
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const ParamHiresStrength = () => {
  const { hiresFix, hiresStrength } = useAppSelector(hiresStrengthSelector);

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const handleHiresStrength = (v: number) => {
    dispatch(setHiresStrength(v));
  };

  const handleHiResStrengthReset = () => {
    dispatch(setHiresStrength(0.75));
  };

  return (
    <IAISlider
      label={t('parameters.hiresStrength')}
      step={0.01}
      min={0.01}
      max={0.99}
      onChange={handleHiresStrength}
      value={hiresStrength}
      isInteger={false}
      withInput
      withSliderMarks
      // inputWidth={22}
      withReset
      handleReset={handleHiResStrengthReset}
      isDisabled={!hiresFix}
    />
  );
};
