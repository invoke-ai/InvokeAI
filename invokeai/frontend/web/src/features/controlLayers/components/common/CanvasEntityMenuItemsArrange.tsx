import { MenuItem } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  entityArrangedBackwardOne,
  entityArrangedForwardOne,
  entityArrangedToBack,
  entityArrangedToFront,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasEntityIdentifier, CanvasV2State } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowDownBold, PiArrowLineDownBold, PiArrowLineUpBold, PiArrowUpBold } from 'react-icons/pi';

const getIndexAndCount = (
  canvasV2: CanvasV2State,
  { id, type }: CanvasEntityIdentifier
): { index: number; count: number } => {
  if (type === 'raster_layer') {
    return {
      index: canvasV2.rasterLayers.entities.findIndex((entity) => entity.id === id),
      count: canvasV2.rasterLayers.entities.length,
    };
  } else if (type === 'control_layer') {
    return {
      index: canvasV2.controlLayers.entities.findIndex((entity) => entity.id === id),
      count: canvasV2.controlLayers.entities.length,
    };
  } else if (type === 'regional_guidance') {
    return {
      index: canvasV2.regions.entities.findIndex((entity) => entity.id === id),
      count: canvasV2.regions.entities.length,
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
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
        const { index, count } = getIndexAndCount(canvasV2, entityIdentifier);
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
      <MenuItem onClick={moveToFront} isDisabled={!validActions.canMoveToFront} icon={<PiArrowLineUpBold />}>
        {t('controlLayers.moveToFront')}
      </MenuItem>
      <MenuItem onClick={moveForwardOne} isDisabled={!validActions.canMoveForwardOne} icon={<PiArrowUpBold />}>
        {t('controlLayers.moveForward')}
      </MenuItem>
      <MenuItem onClick={moveBackwardOne} isDisabled={!validActions.canMoveBackwardOne} icon={<PiArrowDownBold />}>
        {t('controlLayers.moveBackward')}
      </MenuItem>
      <MenuItem onClick={moveToBack} isDisabled={!validActions.canMoveToBack} icon={<PiArrowLineDownBold />}>
        {t('controlLayers.moveToBack')}
      </MenuItem>
    </>
  );
});

CanvasEntityMenuItemsArrange.displayName = 'CanvasEntityArrangeMenuItems';
