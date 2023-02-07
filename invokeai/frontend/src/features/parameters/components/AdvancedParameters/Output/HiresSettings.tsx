import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import IAISwitch from 'common/components/IAISwitch';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import {
  setHiresFix,
  setHiresStrength,
} from 'features/parameters/store/postprocessingSlice';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';
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

const HiresStrength = () => {
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
      label={t('parameters:hiresStrength')}
      step={0.01}
      min={0.01}
      max={0.99}
      onChange={handleHiresStrength}
      value={hiresStrength}
      isInteger={false}
      withInput
      withSliderMarks
      inputWidth={'5.5rem'}
      withReset
      handleReset={handleHiResStrengthReset}
      isSliderDisabled={!hiresFix}
      isInputDisabled={!hiresFix}
      isResetDisabled={!hiresFix}
    />
  );
};

/**
 * Hires Fix Toggle
 */
const HiresSettings = () => {
  const dispatch = useAppDispatch();

  const hiresFix = useAppSelector(
    (state: RootState) => state.postprocessing.hiresFix
  );

  const { t } = useTranslation();

  const handleChangeHiresFix = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHiresFix(e.target.checked));

  return (
    <Flex gap={2} direction={'column'}>
      <IAISwitch
        label={t('parameters:hiresOptim')}
        fontSize={'md'}
        isChecked={hiresFix}
        onChange={handleChangeHiresFix}
      />
      <HiresStrength />
    </Flex>
  );
};

export default HiresSettings;
