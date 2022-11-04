import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import React from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAINumberInput from 'common/components/IAINumberInput';
import { mayGenerateMultipleImagesSelector } from 'features/options/optionsSelectors';
import { OptionsState, setIterations } from 'features/options/optionsSlice';
import { inputWidth } from './MainOptions';

const mainIterationsSelector = createSelector(
  [(state: RootState) => state.options, mayGenerateMultipleImagesSelector],
  (options: OptionsState, mayGenerateMultipleImages) => {
    const { iterations } = options;

    return {
      iterations,
      mayGenerateMultipleImages,
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
  const { iterations, mayGenerateMultipleImages } = useAppSelector(
    mainIterationsSelector
  );

  const handleChangeIterations = (v: number) => dispatch(setIterations(v));

  return (
    <IAINumberInput
      label="Images"
      step={1}
      min={1}
      max={9999}
      isDisabled={!mayGenerateMultipleImages}
      onChange={handleChangeIterations}
      value={iterations}
      width={inputWidth}
      labelFontSize={0.5}
      styleClass="main-option-block"
      textAlign="center"
    />
  );
}
