import { MenuItem } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import { controlLayerWithTransparencyEffectToggled } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfBold } from 'react-icons/pi';

const buildSelectWithTransparencyEffect = (entityIdentifier: CanvasEntityIdentifier<'control_layer'>) =>
  createSelector(
    selectCanvasSlice,
    (canvas) =>
      selectEntityOrThrow(canvas, entityIdentifier, 'ControlLayerMenuItemsTransparencyEffect').withTransparencyEffect
  );

export const ControlLayerMenuItemsTransparencyEffect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const isLocked = useEntityIsLocked(entityIdentifier);
  const selectWithTransparencyEffect = useMemo(
    () => buildSelectWithTransparencyEffect(entityIdentifier),
    [entityIdentifier]
  );
  const withTransparencyEffect = useAppSelector(selectWithTransparencyEffect);
  const onToggle = useCallback(() => {
    dispatch(controlLayerWithTransparencyEffectToggled({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem onClick={onToggle} icon={<PiDropHalfBold />} isDisabled={isLocked}>
      {withTransparencyEffect
        ? t('controlLayers.disableTransparencyEffect')
        : t('controlLayers.enableTransparencyEffect')}
    </MenuItem>
  );
});

ControlLayerMenuItemsTransparencyEffect.displayName = 'ControlLayerMenuItemsTransparencyEffect';
