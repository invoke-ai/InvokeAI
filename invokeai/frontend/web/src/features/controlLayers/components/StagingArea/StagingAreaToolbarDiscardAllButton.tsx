import { IconButton } from '@invoke-ai/ui-library';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useDeleteQueueItemsByDestinationMutation } from 'services/api/endpoints/queue';

export const StagingAreaToolbarDiscardAllButton = memo(() => {
  const ctx = useCanvasSessionContext();
  const { t } = useTranslation();
  const [deleteByDestination] = useDeleteQueueItemsByDestinationMutation();

  const discardAll = useCallback(() => {
    deleteByDestination({ destination: ctx.session.id });
  }, [deleteByDestination, ctx.session.id]);

  return (
    <IconButton
      tooltip={`${t('controlLayers.stagingArea.discardAll')} (Esc)`}
      aria-label={t('controlLayers.stagingArea.discardAll')}
      icon={<PiTrashSimpleBold />}
      onClick={discardAll}
      colorScheme="error"
      fontSize={16}
    />
  );
});

StagingAreaToolbarDiscardAllButton.displayName = 'StagingAreaToolbarDiscardAllButton';
