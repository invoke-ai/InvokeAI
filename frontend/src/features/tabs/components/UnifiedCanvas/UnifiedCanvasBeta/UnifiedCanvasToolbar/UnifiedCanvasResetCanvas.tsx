import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  resetCanvas,
  resizeAndScaleCanvas,
} from 'features/canvas/store/canvasSlice';
import React from 'react';
import { FaTrash } from 'react-icons/fa';

export default function UnifiedCanvasResetCanvas() {
  const dispatch = useAppDispatch();
  const isStaging = useAppSelector(isStagingSelector);

  const handleResetCanvas = () => {
    dispatch(resetCanvas());
    dispatch(resizeAndScaleCanvas());
  };
  return (
    <IAIIconButton
      aria-label="Clear Canvas"
      tooltip="Clear Canvas"
      icon={<FaTrash />}
      onClick={handleResetCanvas}
      style={{ backgroundColor: 'var(--btn-delete-image)' }}
      isDisabled={isStaging}
    />
  );
}
