import { useAppDispatch } from 'app/store';
import IAIButton from 'common/components/IAIButton';

import { clearMask } from 'features/canvas/store/canvasSlice';
import React from 'react';

import { FaTrash } from 'react-icons/fa';

export default function UnifiedCanvasClearMask() {
  const dispatch = useAppDispatch();

  const handleClearMask = () => dispatch(clearMask());

  return (
    <IAIButton
      size={'sm'}
      leftIcon={<FaTrash />}
      onClick={handleClearMask}
      tooltip="Clear Mask (Shift+C)"
    >
      Clear
    </IAIButton>
  );
}
