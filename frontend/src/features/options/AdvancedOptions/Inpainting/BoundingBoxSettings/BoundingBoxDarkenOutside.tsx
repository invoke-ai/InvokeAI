import React from 'react';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import {
  canvasSelector,
  setShouldDarkenOutsideBoundingBox,
} from 'features/canvas/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';

const selector = createSelector(
  canvasSelector,
  (canvas) => canvas.shouldDarkenOutsideBoundingBox
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
