import { Menu, MenuDivider, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityActionMenuItems } from 'features/controlLayers/components/common/CanvasEntityActionMenuItems';
import { CanvasEntityMenuButton } from 'features/controlLayers/components/common/CanvasEntityMenuButton';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useAddIPAdapterToRGLayer } from 'features/controlLayers/hooks/addLayerHooks';
import {
  rgNegativePromptChanged,
  rgPositivePromptChanged,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import { selectRGOrThrow } from 'features/controlLayers/store/regionsReducers';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const RGActionsMenu = memo(() => {
  const { id } = useEntityIdentifierContext();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [onAddIPAdapter, isAddIPAdapterDisabled] = useAddIPAdapterToRGLayer(id);
  const selectActionsValidity = useMemo(
    () =>
      createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
        const rg = selectRGOrThrow(canvasV2, id);
        return {
          isAddPositivePromptDisabled: rg.positivePrompt === null,
          isAddNegativePromptDisabled: rg.negativePrompt === null,
        };
      }),
    [id]
  );
  const actions = useAppSelector(selectActionsValidity);
  const onAddPositivePrompt = useCallback(() => {
    dispatch(rgPositivePromptChanged({ id: id, prompt: '' }));
  }, [dispatch, id]);
  const onAddNegativePrompt = useCallback(() => {
    dispatch(rgNegativePromptChanged({ id: id, prompt: '' }));
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
        <CanvasEntityActionMenuItems />
      </MenuList>
    </Menu>
  );
});

RGActionsMenu.displayName = 'RGActionsMenu';
