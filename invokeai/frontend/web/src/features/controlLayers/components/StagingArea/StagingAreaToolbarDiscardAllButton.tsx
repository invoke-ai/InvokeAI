import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { stagingAreaReset } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardAllButton = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const discardAll = useCallback(() => {
    dispatch(stagingAreaReset());
  }, [dispatch]);

  return (
    <IconButton
      tooltip={`${t('unifiedCanvas.discardAll')} (Esc)`}
      aria-label={t('unifiedCanvas.discardAll')}
      icon={<PiTrashSimpleBold />}
      onClick={discardAll}
      colorScheme="error"
      fontSize={16}
    />
  );
});

StagingAreaToolbarDiscardAllButton.displayName = 'StagingAreaToolbarDiscardAllButton';
