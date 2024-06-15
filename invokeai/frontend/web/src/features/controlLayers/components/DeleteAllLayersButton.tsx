import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { caAllDeleted } from 'features/controlLayers/store/controlAdaptersSlice';
import { ipaAllDeleted } from 'features/controlLayers/store/ipAdaptersSlice';
import { layerAllDeleted } from 'features/controlLayers/store/layersSlice';
import { rgAllDeleted } from 'features/controlLayers/store/regionalGuidanceSlice';
import { selectEntityCount } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const DeleteAllLayersButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityCount = useAppSelector(selectEntityCount);
  const onClick = useCallback(() => {
    dispatch(caAllDeleted());
    dispatch(rgAllDeleted());
    dispatch(ipaAllDeleted());
    dispatch(layerAllDeleted());
  }, [dispatch]);

  return (
    <Button
      onClick={onClick}
      leftIcon={<PiTrashSimpleBold />}
      variant="ghost"
      colorScheme="error"
      isDisabled={entityCount > 0}
      data-testid="control-layers-delete-all-layers-button"
    >
      {t('controlLayers.deleteAll')}
    </Button>
  );
});

DeleteAllLayersButton.displayName = 'DeleteAllLayersButton';
