import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { entityReset } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsReset = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();

  const resetEntity = useCallback(() => {
    dispatch(entityReset({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem onClick={resetEntity} icon={<PiArrowCounterClockwiseBold />}>
      {t('accessibility.reset')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsReset.displayName = 'CanvasEntityMenuItemsReset';
