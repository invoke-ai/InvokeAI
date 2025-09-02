import { IconButton } from '@invoke-ai/ui-library';
import { useCanvasManagerSafe } from 'features/controlLayers/hooks/useCanvasManager';
import { useCanvasIsBusySafe } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useVisibleEntityCountByType } from 'features/controlLayers/hooks/useVisibleEntityCountByType';
import type { CanvasEntityType } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStackBold } from 'react-icons/pi';

type Props = {
  type: CanvasEntityType;
};

export const CanvasEntityMergeVisibleButton = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManagerSafe();
  const isBusy = useCanvasIsBusySafe();
  const entityCount = useVisibleEntityCountByType(type);
  const mergeVisible = useCallback(() => {
    if (canvasManager) {
      canvasManager.compositor.mergeVisibleOfType(type);
    }
  }, [canvasManager, type]);

  return (
    <IconButton
      size="sm"
      aria-label={t('controlLayers.mergeVisible')}
      tooltip={t('controlLayers.mergeVisible')}
      variant="link"
      icon={<PiStackBold />}
      onClick={mergeVisible}
      alignSelf="stretch"
      isDisabled={entityCount <= 1 || isBusy || !canvasManager}
    />
  );
});

CanvasEntityMergeVisibleButton.displayName = 'CanvasEntityMergeVisibleButton';
