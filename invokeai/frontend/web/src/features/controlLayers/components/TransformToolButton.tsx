import { Button, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { toolIsTransformingChanged } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiResizeBold } from 'react-icons/pi';

export const TransformToolButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isTransforming = useAppSelector((s) => s.canvasV2.tool.isTransforming);
  const isDisabled = useAppSelector(
    (s) => s.canvasV2.selectedEntityIdentifier === null || s.canvasV2.session.isStaging
  );

  const onTransform = useCallback(() => {
    dispatch(toolIsTransformingChanged(true));
  }, [dispatch]);

  const onApplyTransformation = useCallback(() => {
    false && dispatch(toolIsTransformingChanged(true));
  }, [dispatch]);

  const onCancelTransformation = useCallback(() => {
    dispatch(toolIsTransformingChanged(false));
  }, [dispatch]);

  useHotkeys(['ctrl+t', 'meta+t'], onTransform, { enabled: !isDisabled }, [isDisabled, onTransform]);

  if (isTransforming) {
    return (
      <>
        <Button onClick={onApplyTransformation}>Apply</Button>
        <Button onClick={onCancelTransformation}>Cancel</Button>
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
