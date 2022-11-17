import React from 'react';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import {
  canvasSelector,
  setShouldLockBoundingBox,
} from 'features/canvas/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';

const boundingBoxLockSelector = createSelector(
  canvasSelector,
  (canvas) => canvas.shouldLockBoundingBox
);

export default function BoundingBoxLock() {
  const shouldLockBoundingBox = useAppSelector(boundingBoxLockSelector);
  const dispatch = useAppDispatch();

  const handleChangeShouldLockBoundingBox = () => {
    dispatch(setShouldLockBoundingBox(!shouldLockBoundingBox));
  };
  return (
    <IAICheckbox
      label="Lock Bounding Box"
      isChecked={shouldLockBoundingBox}
      onChange={handleChangeShouldLockBoundingBox}
      styleClass="inpainting-bounding-box-darken"
    />
  );
}
