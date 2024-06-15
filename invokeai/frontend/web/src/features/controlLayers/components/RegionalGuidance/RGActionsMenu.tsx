import { Menu, MenuDivider, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createMemoizedAppSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityMenuButton } from 'features/controlLayers/components/common/CanvasEntityMenuButton';
import { useAddIPAdapterToRGLayer } from 'features/controlLayers/hooks/addLayerHooks';
import {
  rgDeleted,
  rgMovedBackwardOne,
  rgMovedForwardOne,
  rgMovedToBack,
  rgMovedToFront,
  rgNegativePromptChanged,
  rgPositivePromptChanged,
  rgReset,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import { selectRGOrThrow } from 'features/controlLayers/store/regionsReducers';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowCounterClockwiseBold,
  PiArrowDownBold,
  PiArrowLineDownBold,
  PiArrowLineUpBold,
  PiArrowUpBold,
  PiPlusBold,
  PiTrashSimpleBold,
} from 'react-icons/pi';

type Props = {
  id: string;
};

const selectActionsValidity = createMemoizedAppSelector(
  [selectCanvasV2Slice, (canvasV2, id: string) => id],
  (canvasV2, id) => {
    const rg = selectRGOrThrow(canvasV2, id);
    const rgIndex = canvasV2.regions.indexOf(rg);
    const rgCount = canvasV2.regions.length;
    return {
      isMoveForwardOneDisabled: rgIndex < rgCount - 1,
      isMoveBackardOneDisabled: rgIndex > 0,
      isMoveToFrontDisabled: rgIndex < rgCount - 1,
      isMoveToBackDisabled: rgIndex > 0,
      isAddPositivePromptDisabled: rg.positivePrompt === null,
      isAddNegativePromptDisabled: rg.negativePrompt === null,
    };
  }
);

export const RGActionsMenu = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [onAddIPAdapter, isAddIPAdapterDisabled] = useAddIPAdapterToRGLayer(id);
  const actions = useAppSelector((s) => selectActionsValidity(s, id));
  const onDelete = useCallback(() => {
    dispatch(rgDeleted({ id }));
  }, [dispatch, id]);
  const onReset = useCallback(() => {
    dispatch(rgReset({ id }));
  }, [dispatch, id]);
  const onMoveForwardOne = useCallback(() => {
    dispatch(rgMovedForwardOne({ id }));
  }, [dispatch, id]);
  const onMoveToFront = useCallback(() => {
    dispatch(rgMovedToFront({ id }));
  }, [dispatch, id]);
  const onMoveBackwardOne = useCallback(() => {
    dispatch(rgMovedBackwardOne({ id }));
  }, [dispatch, id]);
  const onMoveToBack = useCallback(() => {
    dispatch(rgMovedToBack({ id }));
  }, [dispatch, id]);
  const onAddPositivePrompt = useCallback(() => {
    dispatch(rgPositivePromptChanged({ id, prompt: '' }));
  }, [dispatch, id]);
  const onAddNegativePrompt = useCallback(() => {
    dispatch(rgNegativePromptChanged({ id, prompt: '' }));
  }, [dispatch, id]);

  return (
    <Menu>
      <CanvasEntityMenuButton />
      <MenuList>
        <MenuItem onClick={onAddPositivePrompt} isDisabled={actions.isAddPositivePromptDisabled} icon={<PiPlusBold />}>
          {t('controlLayers.addPositivePrompt')}
        </MenuItem>
        <MenuItem onClick={onAddNegativePrompt} isDisabled={actions.isAddNegativePromptDisabled} icon={<PiPlusBold />}>
          {t('controlLayers.addNegativePrompt')}
        </MenuItem>
        <MenuItem onClick={onAddIPAdapter} icon={<PiPlusBold />} isDisabled={isAddIPAdapterDisabled}>
          {t('controlLayers.addIPAdapter')}
        </MenuItem>
        <MenuDivider />
        <MenuItem onClick={onMoveToFront} isDisabled={actions.isMoveToFrontDisabled} icon={<PiArrowLineUpBold />}>
          {t('controlLayers.moveToFront')}
        </MenuItem>
        <MenuItem onClick={onMoveForwardOne} isDisabled={actions.isMoveForwardOneDisabled} icon={<PiArrowUpBold />}>
          {t('controlLayers.moveForward')}
        </MenuItem>
        <MenuItem onClick={onMoveBackwardOne} isDisabled={actions.isMoveBackardOneDisabled} icon={<PiArrowDownBold />}>
          {t('controlLayers.moveBackward')}
        </MenuItem>
        <MenuItem onClick={onMoveToBack} isDisabled={actions.isMoveToBackDisabled} icon={<PiArrowLineDownBold />}>
          {t('controlLayers.moveToBack')}
        </MenuItem>
        <MenuDivider />
        <MenuItem onClick={onReset} icon={<PiArrowCounterClockwiseBold />}>
          {t('accessibility.reset')}
        </MenuItem>
        <MenuItem onClick={onDelete} icon={<PiTrashSimpleBold />} color="error.300">
          {t('common.delete')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

RGActionsMenu.displayName = 'RGActionsMenu';
