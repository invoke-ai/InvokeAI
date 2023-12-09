import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import SubParametersWrapper from 'features/parameters/components/Parameters/SubParametersWrapper';
import { setSDXLImg2ImgDenoisingStrength } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector([stateSelector], ({ sdxl }) => {
  const { sdxlImg2ImgDenoisingStrength } = sdxl;

  return {
    sdxlImg2ImgDenoisingStrength,
  };
});

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
    <IAIInformationalPopover feature="paramDenoisingStrength">
      <SubParametersWrapper>
        <IAISlider
          label={t('sdxl.denoisingStrength')}
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
    </IAIInformationalPopover>
  );
};

export default memo(ParamSDXLImg2ImgDenoisingStrength);
