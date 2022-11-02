import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import React from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAINumberInput from '../../../common/components/IAINumberInput';
import { mayGenerateMultipleImagesSelector } from '../optionsSelectors';
import { OptionsState, setIterations } from '../optionsSlice';
import { fontSize, inputWidth } from './MainOptions';

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
      fontSize={fontSize}
      styleClass="main-option-block"
      textAlign="center"
    />
  );
}
