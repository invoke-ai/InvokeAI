import { Menu, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityMenuButton } from 'features/controlLayers/components/common/CanvasEntityMenuButton';
import {
  caDeleted,
  caMovedBackwardOne,
  caMovedForwardOne,
  caMovedToBack,
  caMovedToFront,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import { selectCAOrThrow } from 'features/controlLayers/store/controlAdaptersReducers';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowDownBold,
  PiArrowLineDownBold,
  PiArrowLineUpBold,
  PiArrowUpBold,
  PiTrashSimpleBold,
} from 'react-icons/pi';

type Props = {
  id: string;
};

export const CAActionsMenu = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
        const ca = selectCAOrThrow(canvasV2, id);
        const caIndex = canvasV2.controlAdapters.indexOf(ca);
        const caCount = canvasV2.controlAdapters.length;
        return {
          canMoveForward: caIndex < caCount - 1,
          canMoveBackward: caIndex > 0,
          canMoveToFront: caIndex < caCount - 1,
          canMoveToBack: caIndex > 0,
        };
      }),
    [id]
  );
  const validActions = useAppSelector(selectValidActions);
  const onDelete = useCallback(() => {
    dispatch(caDeleted({ id }));
  }, [dispatch, id]);
  const moveForwardOne = useCallback(() => {
    dispatch(caMovedForwardOne({ id }));
  }, [dispatch, id]);
  const moveToFront = useCallback(() => {
    dispatch(caMovedToFront({ id }));
  }, [dispatch, id]);
  const moveBackwardOne = useCallback(() => {
    dispatch(caMovedBackwardOne({ id }));
  }, [dispatch, id]);
  const moveToBack = useCallback(() => {
    dispatch(caMovedToBack({ id }));
  }, [dispatch, id]);

  return (
    <Menu>
      <CanvasEntityMenuButton />
      <MenuList>
        <MenuItem onClick={moveToFront} isDisabled={!validActions.canMoveToFront} icon={<PiArrowLineUpBold />}>
          {t('controlLayers.moveToFront')}
        </MenuItem>
        <MenuItem onClick={moveForwardOne} isDisabled={!validActions.canMoveForward} icon={<PiArrowUpBold />}>
          {t('controlLayers.moveForward')}
        </MenuItem>
        <MenuItem onClick={moveBackwardOne} isDisabled={!validActions.canMoveBackward} icon={<PiArrowDownBold />}>
          {t('controlLayers.moveBackward')}
        </MenuItem>
        <MenuItem onClick={moveToBack} isDisabled={!validActions.canMoveToBack} icon={<PiArrowLineDownBold />}>
          {t('controlLayers.moveToBack')}
        </MenuItem>
        <MenuItem onClick={onDelete} icon={<PiTrashSimpleBold />} color="error.300">
          {t('common.delete')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

CAActionsMenu.displayName = 'CAActionsMenu';
