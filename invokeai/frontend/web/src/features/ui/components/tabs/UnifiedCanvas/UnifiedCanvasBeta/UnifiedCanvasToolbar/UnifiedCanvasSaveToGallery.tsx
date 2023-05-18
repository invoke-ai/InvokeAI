import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { canvasSavedToGallery } from 'features/canvas/store/actions';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaSave } from 'react-icons/fa';

export default function UnifiedCanvasSaveToGallery() {
  const isStaging = useAppSelector(isStagingSelector);
  const canvasBaseLayer = getCanvasBaseLayer();
  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  useHotkeys(
    ['shift+s'],
    () => {
      handleSaveToGallery();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  );

  const handleSaveToGallery = () => {
    dispatch(canvasSavedToGallery());
  };

  return (
    <IAIIconButton
      aria-label={`${t('unifiedCanvas.saveToGallery')} (Shift+S)`}
      tooltip={`${t('unifiedCanvas.saveToGallery')} (Shift+S)`}
      icon={<FaSave />}
      onClick={handleSaveToGallery}
      isDisabled={isStaging}
    />
  );
}
