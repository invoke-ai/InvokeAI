import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { mergeAndUploadCanvas } from 'features/canvas/store/thunks/mergeAndUploadCanvas';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaDownload } from 'react-icons/fa';

export default function UnifiedCanvasDownloadImage() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const canvasBaseLayer = getCanvasBaseLayer();

  const isStaging = useAppSelector(isStagingSelector);

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const shouldCropToBoundingBoxOnSave = useAppSelector(
    (state: RootState) => state.canvas.shouldCropToBoundingBoxOnSave
  );

  useHotkeys(
    ['shift+d'],
    () => {
      handleDownloadAsImage();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  );

  const handleDownloadAsImage = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
        cropToBoundingBox: shouldCropToBoundingBoxOnSave,
        shouldDownload: true,
      })
    );
  };
  return (
    <IAIIconButton
      aria-label={`${t('unifiedCanvas.downloadAsImage')} (Shift+D)`}
      tooltip={`${t('unifiedCanvas.downloadAsImage')} (Shift+D)`}
      icon={<FaDownload />}
      onClick={handleDownloadAsImage}
      isDisabled={isStaging}
    />
  );
}
