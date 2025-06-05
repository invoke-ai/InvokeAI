import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useDeleteQueueItemMutation } from 'services/api/endpoints/queue';

export const StagingAreaToolbarDiscardSelectedButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const ctx = useCanvasSessionContext();
  const [deleteQueueItem] = useDeleteQueueItemMutation();
  const selectedItemId = useStore(ctx.$selectedItemId);

  const { t } = useTranslation();

  const discardSelected = useCallback(() => {
    if (selectedItemId === null) {
      return;
    }
    deleteQueueItem({ item_id: selectedItemId });
  }, [selectedItemId, deleteQueueItem]);

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.discard')}
      aria-label={t('controlLayers.stagingArea.discard')}
      icon={<PiXBold />}
      onClick={discardSelected}
      colorScheme="invokeBlue"
      fontSize={16}
      isDisabled={selectedItemId === null || isDisabled}
    />
  );
});

StagingAreaToolbarDiscardSelectedButton.displayName = 'StagingAreaToolbarDiscardSelectedButton';
