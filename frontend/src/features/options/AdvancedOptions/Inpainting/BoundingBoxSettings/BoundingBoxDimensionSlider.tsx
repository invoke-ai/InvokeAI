import React from 'react';
import IAISlider from 'common/components/IAISlider';

import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { createSelector } from '@reduxjs/toolkit';
import {
  currentCanvasSelector,
  GenericCanvasState,
  // InpaintingState,
  setBoundingBoxDimensions,
} from 'features/canvas/canvasSlice';

import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import _ from 'lodash';

const boundingBoxDimensionsSelector = createSelector(
  currentCanvasSelector,
  (currentCanvas: GenericCanvasState) => {
    const { stageDimensions, boundingBoxDimensions, shouldLockBoundingBox } =
      currentCanvas;
    return {
      stageDimensions,
      boundingBoxDimensions,
      shouldLockBoundingBox,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

type BoundingBoxDimensionSlidersType = {
  dimension: 'width' | 'height';
  label: string;
};

export default function BoundingBoxDimensionSlider(
  props: BoundingBoxDimensionSlidersType
) {
  const { dimension, label } = props;
  const dispatch = useAppDispatch();
  const { shouldLockBoundingBox, stageDimensions, boundingBoxDimensions } =
    useAppSelector(boundingBoxDimensionsSelector);

  const canvasDimension = stageDimensions[dimension];
  const boundingBoxDimension = boundingBoxDimensions[dimension];

  const handleBoundingBoxDimension = (v: number) => {
    if (dimension == 'width') {
      dispatch(
        setBoundingBoxDimensions({
          ...boundingBoxDimensions,
          width: Math.floor(v),
        })
      );
    }

    if (dimension == 'height') {
      dispatch(
        setBoundingBoxDimensions({
          ...boundingBoxDimensions,
          height: Math.floor(v),
        })
      );
    }
  };

  const handleResetDimension = () => {
    if (dimension == 'width') {
      dispatch(
        setBoundingBoxDimensions({
          ...boundingBoxDimensions,
          width: Math.floor(canvasDimension),
        })
      );
    }
    if (dimension == 'height') {
      dispatch(
        setBoundingBoxDimensions({
          ...boundingBoxDimensions,
          height: Math.floor(canvasDimension),
        })
      );
    }
  };

  return (
    <IAISlider
      label={label}
      min={64}
      max={roundDownToMultiple(canvasDimension, 64)}
      step={64}
      value={boundingBoxDimension}
      onChange={handleBoundingBoxDimension}
      handleReset={handleResetDimension}
      isSliderDisabled={shouldLockBoundingBox}
      isInputDisabled={shouldLockBoundingBox}
      isResetDisabled={
        shouldLockBoundingBox || canvasDimension === boundingBoxDimension
      }
      withSliderMarks
      withInput
      withReset
    />
  );
}
