import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import SDNumberInput from '../../../../common/components/SDNumberInput';
import { setImg2imgStrength } from '../../optionsSlice';

export default function ImageToImageStrength() {
  const img2imgStrength = useAppSelector(
    (state: RootState) => state.options.img2imgStrength
  );

  const dispatch = useAppDispatch();

  const handleChangeStrength = (v: number) => dispatch(setImg2imgStrength(v));

  return (
    <SDNumberInput
      label="Strength"
      step={0.01}
      min={0}
      max={1}
      onChange={handleChangeStrength}
      value={img2imgStrength}
      width="90px"
      isInteger={false}
    />
  );
}
