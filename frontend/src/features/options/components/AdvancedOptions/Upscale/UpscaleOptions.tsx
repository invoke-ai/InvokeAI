import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';

import {
  setUpscalingLevel,
  setUpscalingStrength,
  UpscalingLevel,
  OptionsState,
} from 'features/options/store/optionsSlice';

import { UPSCALING_LEVELS } from 'app/constants';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { SystemState } from 'features/system/store/systemSlice';
import { ChangeEvent } from 'react';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISelect from 'common/components/IAISelect';
import { useTranslation } from 'react-i18next';

const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      upscalingLevel: options.upscalingLevel,
      upscalingStrength: options.upscalingStrength,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    return {
      isESRGANAvailable: system.isESRGANAvailable,
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
const UpscaleOptions = () => {
  const dispatch = useAppDispatch();
  const { upscalingLevel, upscalingStrength } = useAppSelector(optionsSelector);

  const { isESRGANAvailable } = useAppSelector(systemSelector);

  const { t } = useTranslation();

  const handleChangeLevel = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setUpscalingLevel(Number(e.target.value) as UpscalingLevel));

  const handleChangeStrength = (v: number) => dispatch(setUpscalingStrength(v));

  return (
    <div className="upscale-options">
      <IAISelect
        isDisabled={!isESRGANAvailable}
        label={t('options:scale')}
        value={upscalingLevel}
        onChange={handleChangeLevel}
        validValues={UPSCALING_LEVELS}
      />
      <IAINumberInput
        isDisabled={!isESRGANAvailable}
        label={t('options:strength')}
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

export default UpscaleOptions;
