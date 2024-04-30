import { MenuItem } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isRenderableLayer,
  layerMovedBackward,
  layerMovedForward,
  layerMovedToBack,
  layerMovedToFront,
  selectControlLayersSlice,
} from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowDownBold, PiArrowLineDownBold, PiArrowLineUpBold, PiArrowUpBold } from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = { layerId: string };

export const RPLayerMenuArrangeActions = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectControlLayersSlice, (controlLayers) => {
        const layer = controlLayers.present.layers.find((l) => l.id === layerId);
        assert(isRenderableLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        const layerIndex = controlLayers.present.layers.findIndex((l) => l.id === layerId);
        const layerCount = controlLayers.present.layers.length;
        return {
          canMoveForward: layerIndex < layerCount - 1,
          canMoveBackward: layerIndex > 0,
          canMoveToFront: layerIndex < layerCount - 1,
          canMoveToBack: layerIndex > 0,
        };
      }),
    [layerId]
  );
  const validActions = useAppSelector(selectValidActions);
  const moveForward = useCallback(() => {
    dispatch(layerMovedForward(layerId));
  }, [dispatch, layerId]);
  const moveToFront = useCallback(() => {
    dispatch(layerMovedToFront(layerId));
  }, [dispatch, layerId]);
  const moveBackward = useCallback(() => {
    dispatch(layerMovedBackward(layerId));
  }, [dispatch, layerId]);
  const moveToBack = useCallback(() => {
    dispatch(layerMovedToBack(layerId));
  }, [dispatch, layerId]);
  return (
    <>
      <MenuItem onClick={moveToFront} isDisabled={!validActions.canMoveToFront} icon={<PiArrowLineUpBold />}>
        {t('controlLayers.moveToFront')}
      </MenuItem>
      <MenuItem onClick={moveForward} isDisabled={!validActions.canMoveForward} icon={<PiArrowUpBold />}>
        {t('controlLayers.moveForward')}
      </MenuItem>
      <MenuItem onClick={moveBackward} isDisabled={!validActions.canMoveBackward} icon={<PiArrowDownBold />}>
        {t('controlLayers.moveBackward')}
      </MenuItem>
      <MenuItem onClick={moveToBack} isDisabled={!validActions.canMoveToBack} icon={<PiArrowLineDownBold />}>
        {t('controlLayers.moveToBack')}
      </MenuItem>
    </>
  );
});

RPLayerMenuArrangeActions.displayName = 'RPLayerMenuArrangeActions';
