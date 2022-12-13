import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldPreserveMaskedArea } from 'features/canvas/store/canvasSlice';
import React from 'react';

export default function UnifiedCanvasPreserveMask() {
  const dispatch = useAppDispatch();

  const shouldPreserveMaskedArea = useAppSelector(
    (state: RootState) => state.canvas.shouldPreserveMaskedArea
  );

  return (
    <IAICheckbox
      label="Preserve Masked"
      isChecked={shouldPreserveMaskedArea}
      onChange={(e) => dispatch(setShouldPreserveMaskedArea(e.target.checked))}
    />
  );
}
