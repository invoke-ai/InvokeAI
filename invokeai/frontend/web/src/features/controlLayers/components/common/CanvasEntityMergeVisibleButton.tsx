import { IconButton } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
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
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const entityCount = useVisibleEntityCountByType(type);
  const mergeVisible = useCallback(() => {
    canvasManager.compositor.mergeVisibleOfType(type);
  }, [canvasManager.compositor, type]);

  return (
    <IAITooltip label={t('controlLayers.mergeVisible')}>
      <IconButton
        size="sm"
        aria-label={t('controlLayers.mergeVisible')}
        variant="link"
        icon={<PiStackBold />}
        onClick={mergeVisible}
        alignSelf="stretch"
        isDisabled={entityCount <= 1 || isBusy}
      />
    </IAITooltip>
  );
});

CanvasEntityMergeVisibleButton.displayName = 'CanvasEntityMergeVisibleButton';
