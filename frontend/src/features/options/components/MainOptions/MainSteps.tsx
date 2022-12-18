import React from 'react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import { setSteps } from 'features/options/store/optionsSlice';

export default function MainSteps() {
  const dispatch = useAppDispatch();
  const steps = useAppSelector((state: RootState) => state.options.steps);

  const handleChangeSteps = (v: number) => dispatch(setSteps(v));

  return (
    <IAINumberInput
      label="Steps"
      min={1}
      max={9999}
      step={1}
      onChange={handleChangeSteps}
      value={steps}
      width="auto"
      styleClass="main-option-block"
      textAlign="center"
    />
  );
}
