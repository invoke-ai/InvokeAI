import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import SubParametersWrapper from 'features/parameters/components/Parameters/SubParametersWrapper';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { setSDXLImg2ImgDenoisingStrength } from '../store/sdxlSlice';

const selector = createSelector(
  [stateSelector],
  ({ sdxl }) => {
    const { sdxlImg2ImgDenoisingStrength } = sdxl;

    return {
      sdxlImg2ImgDenoisingStrength,
    };
  },
  defaultSelectorOptions
);

const ParamSDXLImg2ImgDenoisingStrength = () => {
  const { sdxlImg2ImgDenoisingStrength } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setSDXLImg2ImgDenoisingStrength(v)),
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(setSDXLImg2ImgDenoisingStrength(0.7));
  }, [dispatch]);

  return (
    <SubParametersWrapper>
      <IAISlider
        label={`${t('parameters.denoisingStrength')}`}
        step={0.01}
        min={0}
        max={1}
        onChange={handleChange}
        handleReset={handleReset}
        value={sdxlImg2ImgDenoisingStrength}
        isInteger={false}
        withInput
        withSliderMarks
        withReset
      />
    </SubParametersWrapper>
  );
};

export default memo(ParamSDXLImg2ImgDenoisingStrength);
