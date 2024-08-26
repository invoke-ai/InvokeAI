import { MenuItem } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { controlLayerWithTransparencyEffectToggled } from 'features/controlLayers/store/canvasV2Slice';
import { selectCanvasV2Slice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfBold } from 'react-icons/pi';

export const ControlLayerMenuItemsTransparencyEffect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const selectWithTransparencyEffect = useMemo(
    () =>
      createSelector(selectCanvasV2Slice, (canvasV2) => {
        const entity = selectEntityOrThrow(canvasV2, entityIdentifier);
        return entity.withTransparencyEffect;
      }),
    [entityIdentifier]
  );
  const withTransparencyEffect = useAppSelector(selectWithTransparencyEffect);
  const onToggle = useCallback(() => {
    dispatch(controlLayerWithTransparencyEffectToggled({ entityIdentifier }));
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
