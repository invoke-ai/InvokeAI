import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldShowGrid } from 'features/canvas/store/canvasSlice';
import React from 'react';

export default function UnifiedCanvasShowGrid() {
  const shouldShowGrid = useAppSelector(
    (state: RootState) => state.canvas.shouldShowGrid
  );

  const dispatch = useAppDispatch();

  return (
    <IAICheckbox
      label="Show Grid"
      isChecked={shouldShowGrid}
      onChange={(e) => dispatch(setShouldShowGrid(e.target.checked))}
    />
  );
}
