import { IconButton, Menu, MenuButton, MenuDivider, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { guidanceLayerIPAdapterAdded } from 'app/store/middleware/listenerMiddleware/listeners/regionalControlToControlAdapterBridge';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isMaskedGuidanceLayer,
  layerDeleted,
  layerMovedBackward,
  layerMovedForward,
  layerMovedToBack,
  layerMovedToFront,
  layerReset,
  maskLayerNegativePromptChanged,
  maskLayerPositivePromptChanged,
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
  PiPlusBold,
  PiTrashSimpleBold,
} from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = { layerId: string };

export const RPLayerMenu = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isMaskedGuidanceLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        const layerIndex = regionalPrompts.present.layers.findIndex((l) => l.id === layerId);
        const layerCount = regionalPrompts.present.layers.length;
        return {
          canAddPositivePrompt: layer.positivePrompt === null,
          canAddNegativePrompt: layer.negativePrompt === null,
          canMoveForward: layerIndex < layerCount - 1,
          canMoveBackward: layerIndex > 0,
          canMoveToFront: layerIndex < layerCount - 1,
          canMoveToBack: layerIndex > 0,
        };
      }),
    [layerId]
  );
  const validActions = useAppSelector(selectValidActions);
  const addPositivePrompt = useCallback(() => {
    dispatch(maskLayerPositivePromptChanged({ layerId, prompt: '' }));
  }, [dispatch, layerId]);
  const addNegativePrompt = useCallback(() => {
    dispatch(maskLayerNegativePromptChanged({ layerId, prompt: '' }));
  }, [dispatch, layerId]);
  const addIPAdapter = useCallback(() => {
    dispatch(guidanceLayerIPAdapterAdded(layerId));
  }, [dispatch, layerId]);
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
  const resetLayer = useCallback(() => {
    dispatch(layerReset(layerId));
  }, [dispatch, layerId]);
  const deleteLayer = useCallback(() => {
    dispatch(layerDeleted(layerId));
  }, [dispatch, layerId]);
  return (
    <Menu>
      <MenuButton as={IconButton} aria-label="Layer menu" size="sm" icon={<PiDotsThreeVerticalBold />} />
      <MenuList>
        <MenuItem onClick={addPositivePrompt} isDisabled={!validActions.canAddPositivePrompt} icon={<PiPlusBold />}>
          {t('regionalPrompts.addPositivePrompt')}
        </MenuItem>
        <MenuItem onClick={addNegativePrompt} isDisabled={!validActions.canAddNegativePrompt} icon={<PiPlusBold />}>
          {t('regionalPrompts.addNegativePrompt')}
        </MenuItem>
        <MenuItem onClick={addIPAdapter} icon={<PiPlusBold />}>
          {t('regionalPrompts.addIPAdapter')}
        </MenuItem>
        <MenuDivider />
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

RPLayerMenu.displayName = 'RPLayerMenu';
