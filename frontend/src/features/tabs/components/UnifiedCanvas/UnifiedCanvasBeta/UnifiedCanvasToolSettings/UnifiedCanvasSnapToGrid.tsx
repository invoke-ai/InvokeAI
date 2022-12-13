import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldSnapToGrid } from 'features/canvas/store/canvasSlice';
import React, { ChangeEvent } from 'react';

export default function UnifiedCanvasSnapToGrid() {
  const shouldSnapToGrid = useAppSelector(
    (state: RootState) => state.canvas.shouldSnapToGrid
  );

  const dispatch = useAppDispatch();

  const handleChangeShouldSnapToGrid = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldSnapToGrid(e.target.checked));

  return (
    <IAICheckbox
      label="Snap to Grid (N)"
      isChecked={shouldSnapToGrid}
      onChange={handleChangeShouldSnapToGrid}
    />
  );
}
