import { IconButton } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useVisibleEntityCountByType } from 'features/controlLayers/hooks/useVisibleEntityCountByType';
import type { CanvasRenderableEntityType } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStackBold } from 'react-icons/pi';

type Props = {
  type: CanvasRenderableEntityType;
};

export const CanvasEntityMergeVisibleButton = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const entityCount = useVisibleEntityCountByType(type);
  const mergeVisible = useCallback(() => {
    canvasManager.compositor.mergeVisibleOfType(type);
  }, [canvasManager.compositor, type]);

  return (
    <IconButton
      size="sm"
      aria-label={t('controlLayers.mergeVisible')}
      tooltip={t('controlLayers.mergeVisible')}
      variant="link"
      icon={<PiStackBold />}
      onClick={mergeVisible}
      alignSelf="stretch"
      isDisabled={entityCount <= 1 || isBusy}
    />
  );
});

CanvasEntityMergeVisibleButton.displayName = 'CanvasEntityMergeVisibleButton';
