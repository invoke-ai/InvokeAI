import { IconButton, Menu, MenuButton, MenuDivider, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  layerDeleted,
  layerMovedBackward,
  layerMovedForward,
  layerMovedToBack,
  layerMovedToFront,
  rpLayerReset,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowCounterClockwiseBold,
  PiArrowDownBold,
  PiArrowLineDownBold,
  PiArrowLineUpBold,
  PiArrowUpBold,
  PiDotsThreeVerticalBold,
  PiTrashSimpleBold,
} from 'react-icons/pi';

type Props = { id: string };

export const LayerMenu = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layerIndex = regionalPrompts.present.layers.findIndex((l) => l.id === id);
        const layerCount = regionalPrompts.present.layers.length;
        return {
          canMoveForward: layerIndex < layerCount - 1,
          canMoveBackward: layerIndex > 0,
          canMoveToFront: layerIndex < layerCount - 1,
          canMoveToBack: layerIndex > 0,
        };
      }),
    [id]
  );
  const validActions = useAppSelector(selectValidActions);
  const moveForward = useCallback(() => {
    dispatch(layerMovedForward(id));
  }, [dispatch, id]);
  const moveToFront = useCallback(() => {
    dispatch(layerMovedToFront(id));
  }, [dispatch, id]);
  const moveBackward = useCallback(() => {
    dispatch(layerMovedBackward(id));
  }, [dispatch, id]);
  const moveToBack = useCallback(() => {
    dispatch(layerMovedToBack(id));
  }, [dispatch, id]);
  const resetLayer = useCallback(() => {
    dispatch(rpLayerReset(id));
  }, [dispatch, id]);
  const deleteLayer = useCallback(() => {
    dispatch(layerDeleted(id));
  }, [dispatch, id]);
  return (
    <Menu>
      <MenuButton as={IconButton} aria-label="Layer menu" size="sm" icon={<PiDotsThreeVerticalBold />} />
      <MenuList>
        <MenuItem onClick={moveToFront} isDisabled={!validActions.canMoveToFront} icon={<PiArrowLineUpBold />}>
          {t('regionalPrompts.moveToFront')}
        </MenuItem>
        <MenuItem onClick={moveForward} isDisabled={!validActions.canMoveForward} icon={<PiArrowUpBold />}>
          {t('regionalPrompts.moveForward')}
        </MenuItem>
        <MenuItem onClick={moveBackward} isDisabled={!validActions.canMoveBackward} icon={<PiArrowDownBold />}>
          {t('regionalPrompts.moveBackward')}
        </MenuItem>
        <MenuItem onClick={moveToBack} isDisabled={!validActions.canMoveToBack} icon={<PiArrowLineDownBold />}>
          {t('regionalPrompts.moveToBack')}
        </MenuItem>
        <MenuDivider />
        <MenuItem onClick={resetLayer} icon={<PiArrowCounterClockwiseBold />}>
          {t('accessibility.reset')}
        </MenuItem>
        <MenuItem onClick={deleteLayer} icon={<PiTrashSimpleBold />} color="error.300">
          {t('common.delete')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

LayerMenu.displayName = 'LayerMenu';
