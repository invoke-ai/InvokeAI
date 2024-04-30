import { IconButton } from '@invoke-ai/ui-library';
import { guidanceLayerDeleted } from 'app/store/middleware/listenerMiddleware/listeners/regionalControlToControlAdapterBridge';
import { useAppDispatch } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = { layerId: string };

export const RPLayerDeleteButton = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const deleteLayer = useCallback(() => {
    dispatch(guidanceLayerDeleted(layerId));
  }, [dispatch, layerId]);
  return (
    <IconButton
      size="sm"
      colorScheme="error"
      aria-label={t('common.delete')}
      tooltip={t('common.delete')}
      icon={<PiTrashSimpleBold />}
      onClick={deleteLayer}
    />
  );
});

RPLayerDeleteButton.displayName = 'RPLayerDeleteButton';
