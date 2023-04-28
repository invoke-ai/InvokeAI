import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';

import { clearMask } from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';

import { FaTrash } from 'react-icons/fa';

export default function UnifiedCanvasClearMask() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleClearMask = () => dispatch(clearMask());

  return (
    <IAIButton
      size="sm"
      leftIcon={<FaTrash />}
      onClick={handleClearMask}
      tooltip={`${t('unifiedCanvas.clearMask')} (Shift+C)`}
    >
      {t('unifiedCanvas.betaClear')}
    </IAIButton>
  );
}
