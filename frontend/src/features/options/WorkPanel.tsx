import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import React from 'react';
import { RootState, useAppSelector } from '../../app/store';
import MainOptions from './MainOptions/MainOptions';
import OptionsAccordion from './OptionsAccordion';
import { OptionsState } from './optionsSlice';
import ProcessButtons from './ProcessButtons/ProcessButtons';
import PromptInput from './PromptInput/PromptInput';

const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      showAdvancedOptions: options.showAdvancedOptions,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export default function WorkPanel() {
  const { showAdvancedOptions } = useAppSelector(optionsSelector);
  return (
    <div className="app-options">
      <PromptInput />
      <ProcessButtons />
      <MainOptions />
      {showAdvancedOptions ? <OptionsAccordion /> : null}
    </div>
  );
}
