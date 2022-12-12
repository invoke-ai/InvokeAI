import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldSnapToGrid } from 'features/canvas/store/canvasSlice';
import React, { ChangeEvent } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

export default function UnifiedCanvasSnapToGrid() {
  const shouldSnapToGrid = useAppSelector(
    (state: RootState) => state.canvas.shouldSnapToGrid
  );

  const dispatch = useAppDispatch();

  useHotkeys(
    ['n'],
    () => {
      dispatch(setShouldSnapToGrid(!shouldSnapToGrid));
    },
    {
      enabled: true,
      preventDefault: true,
    },
    [shouldSnapToGrid]
  );

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
