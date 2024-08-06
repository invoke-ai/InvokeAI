import { MenuItem } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  entityArrangedBackwardOne,
  entityArrangedForwardOne,
  entityArrangedToBack,
  entityArrangedToFront,
  entityDeleted,
  entityReset,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasEntityIdentifier, CanvasV2State } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowCounterClockwiseBold,
  PiArrowDownBold,
  PiArrowLineDownBold,
  PiArrowLineUpBold,
  PiArrowUpBold,
  PiTrashSimpleBold,
} from 'react-icons/pi';

const getIndexAndCount = (
  canvasV2: CanvasV2State,
  { id, type }: CanvasEntityIdentifier
): { index: number; count: number } => {
  if (type === 'layer') {
    return {
      index: canvasV2.layers.entities.findIndex((entity) => entity.id === id),
      count: canvasV2.layers.entities.length,
    };
  } else if (type === 'control_adapter') {
    return {
      index: canvasV2.controlAdapters.entities.findIndex((entity) => entity.id === id),
      count: canvasV2.controlAdapters.entities.length,
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

export const CanvasEntityActionMenuItems = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
        const { index, count } = getIndexAndCount(canvasV2, entityIdentifier);
        return {
          isArrangeable:
            entityIdentifier.type === 'layer' ||
            entityIdentifier.type === 'control_adapter' ||
            entityIdentifier.type === 'regional_guidance',
          isDeleteable: entityIdentifier.type !== 'inpaint_mask',
          canMoveForwardOne: index < count - 1,
          canMoveBackwardOne: index > 0,
          canMoveToFront: index < count - 1,
          canMoveToBack: index > 0,
        };
      }),
    [entityIdentifier]
  );

  const validActions = useAppSelector(selectValidActions);

  const deleteEntity = useCallback(() => {
    dispatch(entityDeleted({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);
  const resetEntity = useCallback(() => {
    dispatch(entityReset({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);
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
      {validActions.isArrangeable && (
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
      )}
      <MenuItem onClick={resetEntity} icon={<PiArrowCounterClockwiseBold />}>
        {t('accessibility.reset')}
      </MenuItem>
      {validActions.isDeleteable && (
        <MenuItem onClick={deleteEntity} icon={<PiTrashSimpleBold />} color="error.300">
          {t('common.delete')}
        </MenuItem>
      )}
    </>
  );
});

CanvasEntityActionMenuItems.displayName = 'CanvasEntityActionMenuItems';
