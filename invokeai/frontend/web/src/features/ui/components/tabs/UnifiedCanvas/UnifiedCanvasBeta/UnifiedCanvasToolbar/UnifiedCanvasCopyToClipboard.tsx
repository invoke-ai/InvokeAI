import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { canvasCopiedToClipboard } from 'features/canvas/store/actions';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { useCopyImageToClipboard } from 'features/ui/hooks/useCopyImageToClipboard';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCopy } from 'react-icons/fa';

export default function UnifiedCanvasCopyToClipboard() {
  const isStaging = useAppSelector(isStagingSelector);
  const canvasBaseLayer = getCanvasBaseLayer();
  const { isClipboardAPIAvailable } = useCopyImageToClipboard();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  useHotkeys(
    ['meta+c', 'ctrl+c'],
    () => {
      handleCopyImageToClipboard();
    },
    {
      enabled: () => !isStaging && isClipboardAPIAvailable,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing, isClipboardAPIAvailable]
  );

  const handleCopyImageToClipboard = useCallback(() => {
    if (!isClipboardAPIAvailable) {
      return;
    }
    dispatch(canvasCopiedToClipboard());
  }, [dispatch, isClipboardAPIAvailable]);

  if (!isClipboardAPIAvailable) {
    return null;
  }

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
