import { FaTrash } from 'react-icons/fa';
import { useAppDispatch } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { clearImageToInpaint } from 'features/canvas/canvasSlice';

export default function IAICanvasClearImageControl() {
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
