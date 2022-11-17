import React from 'react';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldLockBoundingBox } from 'features/canvas/store/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';

const selector = createSelector(
  canvasSelector,
  (canvas) => canvas.shouldLockBoundingBox
);

export default function BoundingBoxLock() {
  const shouldLockBoundingBox = useAppSelector(selector);
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
