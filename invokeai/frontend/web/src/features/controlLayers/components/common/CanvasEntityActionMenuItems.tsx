import { MenuDivider, MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useLayerUseAsControl } from 'features/controlLayers/hooks/useLayerControlAdapter';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import {
  $filteringEntity,
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
  PiCheckBold,
  PiQuestionMarkBold,
  PiStarHalfBold,
  PiTrashSimpleBold,
  PiXBold,
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
  const canvasManager = useStore($canvasManager);
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const useAsControl = useLayerUseAsControl(entityIdentifier);
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

  const isArrangeable = useMemo(
    () => entityIdentifier.type === 'layer' || entityIdentifier.type === 'regional_guidance',
    [entityIdentifier.type]
  );

  const isDeleteable = useMemo(
    () => entityIdentifier.type === 'layer' || entityIdentifier.type === 'regional_guidance',
    [entityIdentifier.type]
  );
  const isFilterable = useMemo(() => entityIdentifier.type === 'layer', [entityIdentifier.type]);
  const isUseAsControlable = useMemo(() => entityIdentifier.type === 'layer', [entityIdentifier.type]);

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
  const filter = useCallback(() => {
    $filteringEntity.set(entityIdentifier);
  }, [entityIdentifier]);
  const debug = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    const entity = canvasManager.stateApi.getEntity(entityIdentifier);
    if (!entity) {
      return;
    }
    console.debug(entity);
  }, [canvasManager, entityIdentifier]);

  return (
    <>
      {isArrangeable && (
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
      {isFilterable && (
        <MenuItem onClick={filter} icon={<PiStarHalfBold />}>
          {t('common.filter')}
        </MenuItem>
      )}
      {isUseAsControlable && (
        <MenuItem onClick={useAsControl.toggle} icon={useAsControl.hasControlAdapter ? <PiXBold /> : <PiCheckBold />}>
          {useAsControl.hasControlAdapter ? t('common.removeControl') : t('common.useAsControl')}
        </MenuItem>
      )}
      <MenuDivider />
      <MenuItem onClick={resetEntity} icon={<PiArrowCounterClockwiseBold />}>
        {t('accessibility.reset')}
      </MenuItem>
      {isDeleteable && (
        <MenuItem onClick={deleteEntity} icon={<PiTrashSimpleBold />} color="error.300">
          {t('common.delete')}
        </MenuItem>
      )}
      <MenuDivider />
      <MenuItem onClick={debug} icon={<PiQuestionMarkBold />} color="warn.300">
        {t('common.debug')}
      </MenuItem>
    </>
  );
});

CanvasEntityActionMenuItems.displayName = 'CanvasEntityActionMenuItems';
