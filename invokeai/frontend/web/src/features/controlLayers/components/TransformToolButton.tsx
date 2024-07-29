import { Button, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { memo, useCallback, useEffect, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiResizeBold } from 'react-icons/pi';

export const TransformToolButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useStore($canvasManager);
  const [isTransforming, setIsTransforming] = useState(false);
  const isDisabled = useAppSelector(
    (s) => s.canvasV2.selectedEntityIdentifier === null || s.canvasV2.session.isStaging
  );

  useEffect(() => {
    if (!canvasManager) {
      return;
    }
    canvasManager.onTransform = setIsTransforming;
    return () => {
      canvasManager.onTransform = null;
    };
  }, [canvasManager]);

  const onTransform = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    canvasManager.startTransform();
  }, [canvasManager]);

  const onApplyTransformation = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    canvasManager.applyTransform();
  }, [canvasManager]);

  const onCancelTransformation = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    canvasManager.cancelTransform();
  }, [canvasManager]);

  useHotkeys(['ctrl+t', 'meta+t'], onTransform, { enabled: !isDisabled }, [isDisabled, onTransform]);

  if (isTransforming) {
    return (
      <>
        <Button onClick={onApplyTransformation}>{t('common.apply')}</Button>
        <Button onClick={onCancelTransformation}>{t('common.cancel')}</Button>
      </>
    );
  }

  return (
    <IconButton
      aria-label={`${t('unifiedCanvas.transform')} (Ctrl+T)`}
      tooltip={`${t('unifiedCanvas.transform')} (Ctrl+T)`}
      icon={<PiResizeBold />}
      variant="solid"
      onClick={onTransform}
      isDisabled={isDisabled}
    />
  );
});

TransformToolButton.displayName = 'TransformToolButton';
