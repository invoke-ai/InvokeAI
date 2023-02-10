import { useAppDispatch, useAppSelector } from 'app/storeHooks';

import {
  setUpscalingDenoising,
  setUpscalingLevel,
  setUpscalingStrength,
  UpscalingLevel,
} from 'features/parameters/store/postprocessingSlice';

import { createSelector } from '@reduxjs/toolkit';
import { UPSCALING_LEVELS } from 'app/constants';
import IAISelect from 'common/components/IAISelect';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import IAISlider from 'common/components/IAISlider';
import { Flex } from '@chakra-ui/react';

const parametersSelector = createSelector(
  [postprocessingSelector, systemSelector],

  (
    { upscalingLevel, upscalingStrength, upscalingDenoising },
    { isESRGANAvailable }
  ) => {
    return {
      upscalingLevel,
      upscalingDenoising,
      upscalingStrength,
      isESRGANAvailable,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Displays upscaling/ESRGAN options (level and strength).
 */
const UpscaleSettings = () => {
  const dispatch = useAppDispatch();
  const {
    upscalingLevel,
    upscalingStrength,
    upscalingDenoising,
    isESRGANAvailable,
  } = useAppSelector(parametersSelector);

  const { t } = useTranslation();

  const handleChangeLevel = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setUpscalingLevel(Number(e.target.value) as UpscalingLevel));

  const handleChangeStrength = (v: number) => dispatch(setUpscalingStrength(v));

  return (
    <Flex flexDir="column" rowGap="1rem" minWidth="20rem">
      <IAISelect
        isDisabled={!isESRGANAvailable}
        label={t('parameters:scale')}
        value={upscalingLevel}
        onChange={handleChangeLevel}
        validValues={UPSCALING_LEVELS}
      />
      <IAISlider
        label={t('parameters:denoisingStrength')}
        value={upscalingDenoising}
        min={0}
        max={1}
        step={0.01}
        onChange={(v) => {
          dispatch(setUpscalingDenoising(v));
        }}
        handleReset={() => dispatch(setUpscalingDenoising(0.75))}
        withSliderMarks
        withInput
        withReset
        isSliderDisabled={!isESRGANAvailable}
        isInputDisabled={!isESRGANAvailable}
        isResetDisabled={!isESRGANAvailable}
      />
      <IAISlider
        label={`${t('parameters:upscale')} ${t('parameters:strength')}`}
        value={upscalingStrength}
        min={0}
        max={1}
        step={0.05}
        onChange={handleChangeStrength}
        handleReset={() => dispatch(setUpscalingStrength(0.75))}
        withSliderMarks
        withInput
        withReset
        isSliderDisabled={!isESRGANAvailable}
        isInputDisabled={!isESRGANAvailable}
        isResetDisabled={!isESRGANAvailable}
      />
    </Flex>
  );
};

export default UpscaleSettings;
