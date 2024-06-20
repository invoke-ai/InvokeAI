import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { allEntitiesDeleted } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const DeleteAllLayersButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityCount = useAppSelector((s) => {
    return (
      s.canvasV2.regions.entities.length +
      s.canvasV2.controlAdapters.entities.length +
      s.canvasV2.ipAdapters.entities.length +
      s.canvasV2.layers.entities.length
    );
  });
  const onClick = useCallback(() => {
    dispatch(allEntitiesDeleted());
  }, [dispatch]);

  return (
    <Button
      onClick={onClick}
      leftIcon={<PiTrashSimpleBold />}
      variant="ghost"
      colorScheme="error"
      isDisabled={entityCount === 0}
      data-testid="control-layers-delete-all-layers-button"
    >
      {t('controlLayers.deleteAll')}
    </Button>
  );
});

DeleteAllLayersButton.displayName = 'DeleteAllLayersButton';
