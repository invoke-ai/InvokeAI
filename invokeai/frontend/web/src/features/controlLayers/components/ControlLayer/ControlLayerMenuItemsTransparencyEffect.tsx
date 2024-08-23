import { MenuItem } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  controlLayerWithTransparencyEffectToggled,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import { selectControlLayerEntityOrThrow } from 'features/controlLayers/store/controlLayersReducers';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfBold } from 'react-icons/pi';

export const ControlLayerMenuItemsTransparencyEffect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const selectWithTransparencyEffect = useMemo(
    () =>
      createSelector(selectCanvasV2Slice, (canvasV2) => {
        const entity = selectControlLayerEntityOrThrow(canvasV2, entityIdentifier.id);
        return entity.withTransparencyEffect;
      }),
    [entityIdentifier.id]
  );
  const withTransparencyEffect = useAppSelector(selectWithTransparencyEffect);
  const onToggle = useCallback(() => {
    dispatch(controlLayerWithTransparencyEffectToggled({ id: entityIdentifier.id }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem onClick={onToggle} icon={<PiDropHalfBold />}>
      {withTransparencyEffect
        ? t('controlLayers.disableTransparencyEffect')
        : t('controlLayers.enableTransparencyEffect')}
    </MenuItem>
  );
});

ControlLayerMenuItemsTransparencyEffect.displayName = 'ControlLayerMenuItemsTransparencyEffect';
