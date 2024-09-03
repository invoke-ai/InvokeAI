import { MenuItem } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import {
  entityArrangedBackwardOne,
  entityArrangedForwardOne,
  entityArrangedToBack,
  entityArrangedToFront,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier, CanvasState } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowDownBold, PiArrowLineDownBold, PiArrowLineUpBold, PiArrowUpBold } from 'react-icons/pi';

const getIndexAndCount = (
  canvas: CanvasState,
  { id, type }: CanvasEntityIdentifier
): { index: number; count: number } => {
  if (type === 'raster_layer') {
    return {
      index: canvas.rasterLayers.entities.findIndex((entity) => entity.id === id),
      count: canvas.rasterLayers.entities.length,
    };
  } else if (type === 'control_layer') {
    return {
      index: canvas.controlLayers.entities.findIndex((entity) => entity.id === id),
      count: canvas.controlLayers.entities.length,
    };
  } else if (type === 'regional_guidance') {
    return {
      index: canvas.regions.entities.findIndex((entity) => entity.id === id),
      count: canvas.regions.entities.length,
    };
  } else if (type === 'inpaint_mask') {
    return {
      index: canvas.inpaintMasks.entities.findIndex((entity) => entity.id === id),
      count: canvas.inpaintMasks.entities.length,
    };
  } else if (type === 'ip_adapter') {
    return {
      index: canvas.ipAdapters.entities.findIndex((entity) => entity.id === id),
      count: canvas.ipAdapters.entities.length,
    };
  } else {
    return {
      index: -1,
      count: 0,
    };
  }
};

export const CanvasEntityMenuItemsArrange = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const isBusy = useCanvasIsBusy();
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectCanvasSlice, (canvas) => {
        const { index, count } = getIndexAndCount(canvas, entityIdentifier);
        return {
          canMoveForwardOne: index < count - 1,
          canMoveBackwardOne: index > 0,
          canMoveToFront: index < count - 1,
          canMoveToBack: index > 0,
        };
      }),
    [entityIdentifier]
  );

  const validActions = useAppSelector(selectValidActions);

  const moveForwardOne = useCallback(() => {
    dispatch(entityArrangedForwardOne({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);
  const moveToFront = useCallback(() => {
    dispatch(entityArrangedToFront({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);
  const moveBackwardOne = useCallback(() => {
    dispatch(entityArrangedBackwardOne({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);
  const moveToBack = useCallback(() => {
    dispatch(entityArrangedToBack({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <>
      <MenuItem onClick={moveToFront} isDisabled={!validActions.canMoveToFront || isBusy} icon={<PiArrowLineUpBold />}>
        {t('controlLayers.moveToFront')}
      </MenuItem>
      <MenuItem
        onClick={moveForwardOne}
        isDisabled={!validActions.canMoveForwardOne || isBusy}
        icon={<PiArrowUpBold />}
      >
        {t('controlLayers.moveForward')}
      </MenuItem>
      <MenuItem
        onClick={moveBackwardOne}
        isDisabled={!validActions.canMoveBackwardOne || isBusy}
        icon={<PiArrowDownBold />}
      >
        {t('controlLayers.moveBackward')}
      </MenuItem>
      <MenuItem onClick={moveToBack} isDisabled={!validActions.canMoveToBack || isBusy} icon={<PiArrowLineDownBold />}>
        {t('controlLayers.moveToBack')}
      </MenuItem>
    </>
  );
});

CanvasEntityMenuItemsArrange.displayName = 'CanvasEntityArrangeMenuItems';
