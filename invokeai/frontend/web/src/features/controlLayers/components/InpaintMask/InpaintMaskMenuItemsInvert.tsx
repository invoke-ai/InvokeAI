import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { inpaintMaskInverted } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSelectionInverseBold } from 'react-icons/pi';

export const InpaintMaskMenuItemsInvert = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const canvasSlice = useAppSelector(selectCanvasSlice);

  const handleInvertMask = useCallback(() => {
    dispatch(inpaintMaskInverted({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  // Only show if there are objects to invert and we have a valid bounding box
  const entity = canvasSlice.inpaintMasks.entities.find((entity) => entity.id === entityIdentifier.id);
  const hasObjects = entity?.objects.length > 0;
  const hasBbox = canvasSlice.bbox.rect.width > 0 && canvasSlice.bbox.rect.height > 0;

  if (!hasObjects || !hasBbox) {
    return null;
  }

  return (
    <MenuItem onClick={handleInvertMask} icon={<PiSelectionInverseBold />}>
      {t('controlLayers.invertMask')}
    </MenuItem>
  );
});

InpaintMaskMenuItemsInvert.displayName = 'InpaintMaskMenuItemsInvert';
