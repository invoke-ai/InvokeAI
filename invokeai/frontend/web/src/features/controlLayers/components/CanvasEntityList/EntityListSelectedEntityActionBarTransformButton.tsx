import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isTransformableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFrameCornersBold } from 'react-icons/pi';

export const EntityListSelectedEntityActionBarTransformButton = memo(() => {
  const { t } = useTranslation();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const canvasManager = useCanvasManager();

  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    if (!selectedEntityIdentifier) {
      return;
    }
    if (!isTransformableEntityIdentifier(selectedEntityIdentifier)) {
      return;
    }
    const adapter = canvasManager.getAdapter(selectedEntityIdentifier);
    if (!adapter) {
      return;
    }
    adapter.transformer.startTransform();
  }, [canvasManager, selectedEntityIdentifier]);

  if (!selectedEntityIdentifier) {
    return null;
  }

  if (!isTransformableEntityIdentifier(selectedEntityIdentifier)) {
    return null;
  }

  return (
    <IconButton
      onClick={onClick}
      isDisabled={isBusy}
      size="sm"
      variant="link"
      alignSelf="stretch"
      aria-label={t('controlLayers.transform.transform')}
      tooltip={t('controlLayers.transform.transform')}
      icon={<PiFrameCornersBold />}
    />
  );
});

EntityListSelectedEntityActionBarTransformButton.displayName = 'EntityListSelectedEntityActionBarTransformButton';
