import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { mergeAndUploadCanvas } from 'features/canvas/store/thunks/mergeAndUploadCanvas';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCopy } from 'react-icons/fa';

export default function UnifiedCanvasCopyToClipboard() {
  const isStaging = useAppSelector(isStagingSelector);
  const canvasBaseLayer = getCanvasBaseLayer();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const shouldCropToBoundingBoxOnSave = useAppSelector(
    (state: RootState) => state.canvas.shouldCropToBoundingBoxOnSave
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  useHotkeys(
    ['meta+c', 'ctrl+c'],
    () => {
      handleCopyImageToClipboard();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  );

  const handleCopyImageToClipboard = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
        cropToBoundingBox: shouldCropToBoundingBoxOnSave,
        shouldCopy: true,
      })
    );
  };

  return (
    <IAIIconButton
      aria-label={`${t('unifiedCanvas.copyToClipboard')} (Cmd/Ctrl+C)`}
      tooltip={`${t('unifiedCanvas.copyToClipboard')} (Cmd/Ctrl+C)`}
      icon={<FaCopy />}
      onClick={handleCopyImageToClipboard}
      isDisabled={isStaging}
    />
  );
}
