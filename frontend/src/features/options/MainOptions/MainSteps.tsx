import React from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import SDNumberInput from '../../../common/components/SDNumberInput';
import { setSteps } from '../optionsSlice';
import { fontSize, inputWidth } from './MainOptions';

export default function MainSteps() {
  const dispatch = useAppDispatch();
  const steps = useAppSelector((state: RootState) => state.options.steps);

  const handleChangeSteps = (v: number) => dispatch(setSteps(v));

  return (
    <SDNumberInput
      label="Steps"
      min={1}
      max={9999}
      step={1}
      precision={0}
      onChange={handleChangeSteps}
      value={steps}
      width={inputWidth}
      fontSize={fontSize}
      styleClass="main-option-block"
      textAlign="center"
    />
  );
}
