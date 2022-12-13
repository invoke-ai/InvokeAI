import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldRestrictStrokesToBox } from 'features/canvas/store/canvasSlice';
import React from 'react';

export default function UnifiedCanvasLimitStrokesToBox() {
  const dispatch = useAppDispatch();

  const shouldRestrictStrokesToBox = useAppSelector(
    (state: RootState) => state.canvas.shouldRestrictStrokesToBox
  );

  return (
    <IAICheckbox
      label="Limit To Box"
      isChecked={shouldRestrictStrokesToBox}
      onChange={(e) =>
        dispatch(setShouldRestrictStrokesToBox(e.target.checked))
      }
    />
  );
}
