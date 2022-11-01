import React from 'react';
import { FaTrash } from 'react-icons/fa';
import { useAppDispatch } from '../../../../app/store';
import IAIIconButton from '../../../../common/components/IAIIconButton';
import { clearImageToInpaint } from '../inpaintingSlice';

export default function InpaintingClearImageControl() {
  const dispatch = useAppDispatch();

  const handleClearImage = () => {
    dispatch(clearImageToInpaint());
  };

  return (
    <IAIIconButton
      aria-label="Clear Image"
      tooltip="Clear Image"
      icon={<FaTrash size={16} />}
      onClick={handleClearImage}
    />
  );
}
