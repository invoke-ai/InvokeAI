import React from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setShouldDarkenOutsideBoundingBox,
} from 'features/canvas/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';

const selector = createSelector(
  currentCanvasSelector,
  (currentCanvas: GenericCanvasState) =>
    currentCanvas.shouldDarkenOutsideBoundingBox
);

export default function BoundingBoxDarkenOutside() {
  const dispatch = useAppDispatch();
  const shouldDarkenOutsideBoundingBox = useAppSelector(selector);

  const handleChangeShouldShowBoundingBoxFill = () => {
    dispatch(
      setShouldDarkenOutsideBoundingBox(!shouldDarkenOutsideBoundingBox)
    );
  };

  return (
    <IAICheckbox
      label="Darken Outside Box"
      isChecked={shouldDarkenOutsideBoundingBox}
      onChange={handleChangeShouldShowBoundingBoxFill}
      styleClass="inpainting-bounding-box-darken"
    />
  );
}
