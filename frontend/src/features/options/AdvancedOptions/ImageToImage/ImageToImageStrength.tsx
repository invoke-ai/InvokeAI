import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAISlider from '../../../../common/components/IAISlider';
import { setImg2imgStrength } from '../../optionsSlice';

interface ImageToImageStrengthProps {
  label?: string;
  styleClass?: string;
}

export default function ImageToImageStrength(props: ImageToImageStrengthProps) {
  const { label = 'Strength', styleClass } = props;
  const img2imgStrength = useAppSelector(
    (state: RootState) => state.options.img2imgStrength
  );

  const dispatch = useAppDispatch();

  const handleChangeStrength = (v: number) => dispatch(setImg2imgStrength(v));

  const handleChangeStrengthReset = () => {
    dispatch(setImg2imgStrength(0.5));
  };

  return (
    <IAISlider
      label={label}
      step={0.01}
      min={0.01}
      max={0.99}
      value={img2imgStrength}
      onChange={handleChangeStrength}
      handleReset={handleChangeStrengthReset}
      withSliderMarks
      withReset
      withInput
      inputWidth={'5.5rem'}
      styleClass={styleClass}
    />
  );
}
