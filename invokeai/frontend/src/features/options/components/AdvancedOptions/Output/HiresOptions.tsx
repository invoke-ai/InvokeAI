import { Flex } from '@chakra-ui/react';
import { ChangeEvent } from 'react';
import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import {
  setHiresFix,
  setHiresStrength,
} from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';
import IAISlider from 'common/components/IAISlider';

function HighResStrength() {
  const hiresFix = useAppSelector((state: RootState) => state.options.hiresFix);
  const hiresStrength = useAppSelector(
    (state: RootState) => state.options.hiresStrength
  );

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
      label={t('options:hiresStrength')}
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
}

/**
 * Hires Fix Toggle
 */
const HiresOptions = () => {
  const dispatch = useAppDispatch();

  const hiresFix = useAppSelector((state: RootState) => state.options.hiresFix);

  const { t } = useTranslation();

  const handleChangeHiresFix = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHiresFix(e.target.checked));

  return (
    <Flex gap={2} direction={'column'}>
      <IAISwitch
        label={t('options:hiresOptim')}
        fontSize={'md'}
        isChecked={hiresFix}
        onChange={handleChangeHiresFix}
      />
      <HighResStrength />
    </Flex>
  );
};

export default HiresOptions;
