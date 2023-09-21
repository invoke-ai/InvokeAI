import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { canvasCopiedToClipboard } from 'features/canvas/store/actions';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useCopyImageToClipboard } from 'features/ui/hooks/useCopyImageToClipboard';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCopy } from 'react-icons/fa';

export default function UnifiedCanvasCopyToClipboard() {
  const isStaging = useAppSelector(isStagingSelector);
  const { isClipboardAPIAvailable } = useCopyImageToClipboard();

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
    [isClipboardAPIAvailable]
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
