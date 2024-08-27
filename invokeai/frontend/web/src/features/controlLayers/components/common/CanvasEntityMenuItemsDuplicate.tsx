import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { entityDuplicated } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyFill } from 'react-icons/pi';

export const CanvasEntityMenuItemsDuplicate = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();

  const onClick = useCallback(() => {
    dispatch(entityDuplicated({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem onClick={onClick} icon={<PiCopyFill />}>
      {t('controlLayers.duplicate')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsDuplicate.displayName = 'CanvasEntityMenuItemsDuplicate';
