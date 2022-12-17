import React from 'react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setImg2imgStrength } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

interface ImageToImageStrengthProps {
  label?: string;
  styleClass?: string;
}

export default function ImageToImageStrength(props: ImageToImageStrengthProps) {
  const { t } = useTranslation();
  const { label = `${t('options:strength')}`, styleClass } = props;
  const img2imgStrength = useAppSelector(
    (state: RootState) => state.options.img2imgStrength
  );

  const dispatch = useAppDispatch();

  const handleChangeStrength = (v: number) => dispatch(setImg2imgStrength(v));

  const handleImg2ImgStrengthReset = () => {
    dispatch(setImg2imgStrength(0.75));
  };

  return (
    <IAISlider
      label={label}
      step={0.01}
      min={0.01}
      max={0.99}
      onChange={handleChangeStrength}
      value={img2imgStrength}
      isInteger={false}
      styleClass={styleClass}
      withInput
      withReset
      withSliderMarks
      inputWidth={'5.5rem'}
      handleReset={handleImg2ImgStrengthReset}
    />
  );
}
