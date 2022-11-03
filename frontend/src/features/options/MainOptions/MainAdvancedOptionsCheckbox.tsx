import React, { ChangeEvent } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAICheckbox from '../../../common/components/IAICheckbox';
import { setShowAdvancedOptions } from '../optionsSlice';

export default function MainAdvancedOptionsCheckbox() {
  const showAdvancedOptions = useAppSelector(
    (state: RootState) => state.options.showAdvancedOptions
  );
  const dispatch = useAppDispatch();

  const handleShowAdvancedOptions = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShowAdvancedOptions(e.target.checked));

  return (
    <IAICheckbox
      label="Advanced Options"
      styleClass="advanced-options-checkbox"
      onChange={handleShowAdvancedOptions}
      isChecked={showAdvancedOptions}
    />
  );
}
