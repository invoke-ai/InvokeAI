import React from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setShouldShowBoundingBoxFill,
} from 'features/canvas/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';

const boundingBoxDarkenOutsideSelector = createSelector(
  currentCanvasSelector,
  (currentCanvas: GenericCanvasState) => currentCanvas.shouldShowBoundingBoxFill
);

export default function BoundingBoxDarkenOutside() {
  const dispatch = useAppDispatch();
  const shouldShowBoundingBoxFill = useAppSelector(
    boundingBoxDarkenOutsideSelector
  );

  const handleChangeShouldShowBoundingBoxFill = () => {
    dispatch(setShouldShowBoundingBoxFill(!shouldShowBoundingBoxFill));
  };

  return (
    <IAICheckbox
      label="Darken Outside Box"
      isChecked={shouldShowBoundingBoxFill}
      onChange={handleChangeShouldShowBoundingBoxFill}
      styleClass="inpainting-bounding-box-darken"
    />
  );
}
