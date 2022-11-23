import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import React from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAINumberInput from 'common/components/IAINumberInput';
import {
  OptionsState,
  setIterations,
} from 'features/options/store/optionsSlice';
import { inputWidth } from './MainOptions';

const mainIterationsSelector = createSelector(
  [(state: RootState) => state.options],
  (options: OptionsState) => {
    const { iterations } = options;

    return {
      iterations,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function MainIterations() {
  const dispatch = useAppDispatch();
  const { iterations } = useAppSelector(mainIterationsSelector);

  const handleChangeIterations = (v: number) => dispatch(setIterations(v));

  return (
    <IAINumberInput
      label="Images"
      step={1}
      min={1}
      max={9999}
      onChange={handleChangeIterations}
      value={iterations}
      width={inputWidth}
      labelFontSize={0.5}
      styleClass="main-option-block"
      textAlign="center"
    />
  );
}
