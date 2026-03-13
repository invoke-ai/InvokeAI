import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { IAITooltip } from 'common/components/IAITooltip';
import { useStagingAreaContext } from 'features/controlLayers/components/StagingArea/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCancelQueueItem } from 'features/queue/hooks/useCancelQueueItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardSelectedButton = memo(() => {
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  const ctx = useStagingAreaContext();
  const cancelQueueItem = useCancelQueueItem();
  const discardSelectedIsEnabled = useStore(ctx.$discardSelectedIsEnabled);

  const { t } = useTranslation();

  return (
    <IAITooltip label={t('controlLayers.stagingArea.discard')}>
      <IconButton
        aria-label={t('controlLayers.stagingArea.discard')}
        icon={<PiXBold />}
        onClick={ctx.discardSelected}
        colorScheme="invokeBlue"
        isDisabled={!discardSelectedIsEnabled || cancelQueueItem.isDisabled || !shouldShowStagedImage}
        isLoading={cancelQueueItem.isLoading}
      />
    </IAITooltip>
  );
});

StagingAreaToolbarDiscardSelectedButton.displayName = 'StagingAreaToolbarDiscardSelectedButton';
