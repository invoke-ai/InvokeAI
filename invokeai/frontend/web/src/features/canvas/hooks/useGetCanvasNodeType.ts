import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { canvasSelector } from '../store/canvasSelectors';
import { useMemo } from 'react';
import { getCanvasNodeType } from '../util/getCanvasNodeType';

const selector = createSelector(canvasSelector, (canvas) => {
  const {
    layerState: { objects },
    boundingBoxCoordinates,
    boundingBoxDimensions,
    stageScale,
    isMaskEnabled,
  } = canvas;
  return {
    objects,
    boundingBoxCoordinates,
    boundingBoxDimensions,
    stageScale,
    isMaskEnabled,
  };
});

export const useGetCanvasNodeType = () => {
  const data = useAppSelector(selector);

  const nodeType = useMemo(() => getCanvasNodeType(data), [data]);

  return nodeType;
};
