import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';

export default function UnifiedCanvasResetCanvas() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isStaging = useAppSelector(isStagingSelector);

  const handleResetCanvas = () => {
    dispatch(resetCanvas());
  };
  return (
    <IAIIconButton
      aria-label={t('unifiedCanvas.clearCanvas')}
      tooltip={t('unifiedCanvas.clearCanvas')}
      icon={<FaTrash />}
      onClick={handleResetCanvas}
      isDisabled={isStaging}
      colorScheme="error"
    />
  );
}
