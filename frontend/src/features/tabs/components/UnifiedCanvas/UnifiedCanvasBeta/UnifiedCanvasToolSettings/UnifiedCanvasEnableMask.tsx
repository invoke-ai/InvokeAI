import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAICheckbox from 'common/components/IAICheckbox';
import { setIsMaskEnabled } from 'features/canvas/store/canvasSlice';
import React from 'react';

export default function UnifiedCanvasEnableMask() {
  const isMaskEnabled = useAppSelector(
    (state: RootState) => state.canvas.isMaskEnabled
  );

  const dispatch = useAppDispatch();

  const handleToggleEnableMask = () =>
    dispatch(setIsMaskEnabled(!isMaskEnabled));

  return (
    <IAICheckbox
      label="Enable Mask (H)"
      isChecked={isMaskEnabled}
      onChange={handleToggleEnableMask}
    />
  );
}
