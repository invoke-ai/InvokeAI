import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store';
import _ from 'lodash';
import { currentCanvasSelector, GenericCanvasState } from '../canvasSlice';

const selector = createSelector(
  [currentCanvasSelector],
  (currentCanvas: GenericCanvasState) => {
    return currentCanvas.stageScale;
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const useUnscaleCanvasValue = () => {
  const stageScale = useAppSelector(selector);
  return (value: number) => value / stageScale;
};

export default useUnscaleCanvasValue;
