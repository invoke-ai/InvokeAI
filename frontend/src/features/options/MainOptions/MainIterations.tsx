import React from 'react';
import { useDispatch } from 'react-redux';
import { RootState, useAppSelector } from '../../../app/store';
import SDNumberInput from '../../../common/components/SDNumberInput';
import { setIterations } from '../optionsSlice';
import { fontSize, inputWidth } from './MainOptions';

export default function MainIterations() {
  const dispatch = useDispatch();
  const iterations = useAppSelector(
    (state: RootState) => state.options.iterations
  );

  const handleChangeIterations = (v: number) =>
    dispatch(setIterations(v));

  return (
    <SDNumberInput
      label="Images"
      step={1}
      min={1}
      max={9999}
      precision={0}
      onChange={handleChangeIterations}
      value={iterations}
      width={inputWidth}
      fontSize={fontSize}
      styleClass="main-option-block"
      textAlign="center"
    />
  );
}
