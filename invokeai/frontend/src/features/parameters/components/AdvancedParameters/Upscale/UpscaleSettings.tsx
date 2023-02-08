import { useAppDispatch, useAppSelector } from 'app/storeHooks';

import {
  setUpscalingLevel,
  setUpscalingStrength,
  UpscalingLevel,
} from 'features/parameters/store/postprocessingSlice';

import { createSelector } from '@reduxjs/toolkit';
import { UPSCALING_LEVELS } from 'app/constants';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISelect from 'common/components/IAISelect';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

const parametersSelector = createSelector(
  [postprocessingSelector, systemSelector],

  ({ upscalingLevel, upscalingStrength }, { isESRGANAvailable }) => {
    return {
      upscalingLevel,
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
  const { upscalingLevel, upscalingStrength, isESRGANAvailable } =
    useAppSelector(parametersSelector);

  const { t } = useTranslation();

  const handleChangeLevel = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setUpscalingLevel(Number(e.target.value) as UpscalingLevel));

  const handleChangeStrength = (v: number) => dispatch(setUpscalingStrength(v));

  return (
    <div className="upscale-settings">
      <IAISelect
        isDisabled={!isESRGANAvailable}
        label={t('parameters:scale')}
        value={upscalingLevel}
        onChange={handleChangeLevel}
        validValues={UPSCALING_LEVELS}
      />
      <IAINumberInput
        isDisabled={!isESRGANAvailable}
        label={t('parameters:strength')}
        step={0.05}
        min={0}
        max={1}
        onChange={handleChangeStrength}
        value={upscalingStrength}
        isInteger={false}
      />
    </div>
  );
};

export default UpscaleSettings;
