import { IconButton } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { rasterLayerIsTransparencyLockedToggled } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfBold, PiDropHalfFill } from 'react-icons/pi';

export const RasterLayerIsTransparencyLockedToggle = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext('raster_layer');
  const isBusy = useCanvasIsBusy();
  const dispatch = useAppDispatch();

  const selectIsTransparencyLocked = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        const entity = selectEntity(canvas, entityIdentifier);
        if (!entity) {
          return false;
        }
        return entity.isTransparencyLocked ?? false;
      }),
    [entityIdentifier]
  );

  const isTransparencyLocked = useAppSelector(selectIsTransparencyLocked);

  const onClick = useCallback(() => {
    dispatch(rasterLayerIsTransparencyLockedToggled({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <IconButton
      size="sm"
      aria-label={t(isTransparencyLocked ? 'controlLayers.transparencyLocked' : 'controlLayers.transparencyUnlocked')}
      tooltip={t(isTransparencyLocked ? 'controlLayers.transparencyLocked' : 'controlLayers.transparencyUnlocked')}
      variant="link"
      alignSelf="stretch"
      icon={isTransparencyLocked ? <PiDropHalfFill /> : <PiDropHalfBold />}
      onClick={onClick}
      isDisabled={isBusy}
    />
  );
});

RasterLayerIsTransparencyLockedToggle.displayName = 'RasterLayerIsTransparencyLockedToggle';
