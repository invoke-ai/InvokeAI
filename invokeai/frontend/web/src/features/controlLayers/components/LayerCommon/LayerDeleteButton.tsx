import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { stopPropagation } from 'common/util/stopPropagation';
import { layerDeleted } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = { layerId: string };

export const LayerDeleteButton = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const deleteLayer = useCallback(() => {
    dispatch(layerDeleted(layerId));
  }, [dispatch, layerId]);
  return (
    <IconButton
      size="sm"
      colorScheme="error"
      aria-label={t('common.delete')}
      tooltip={t('common.delete')}
      icon={<PiTrashSimpleBold />}
      onClick={deleteLayer}
      onDoubleClick={stopPropagation} // double click expands the layer
    />
  );
});

LayerDeleteButton.displayName = 'LayerDeleteButton';
