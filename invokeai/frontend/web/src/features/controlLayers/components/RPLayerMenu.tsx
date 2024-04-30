import { IconButton, Menu, MenuButton, MenuDivider, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { RPLayerMenuArrangeActions } from 'features/controlLayers/components/RPLayerMenuArrangeActions';
import { RPLayerMenuMaskedGuidanceActions } from 'features/controlLayers/components/RPLayerMenuMaskedGuidanceActions';
import { useLayerType } from 'features/controlLayers/hooks/layerStateHooks';
import { layerDeleted, layerReset } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiDotsThreeVerticalBold, PiTrashSimpleBold } from 'react-icons/pi';

type Props = { layerId: string };

export const RPLayerMenu = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const layerType = useLayerType(layerId);
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
        {layerType === 'regional_guidance_layer' && (
          <>
            <RPLayerMenuMaskedGuidanceActions layerId={layerId} />
            <MenuDivider />
          </>
        )}
        {(layerType === 'regional_guidance_layer' || layerType === 'control_adapter_layer') && (
          <>
            <RPLayerMenuArrangeActions layerId={layerId} />
            <MenuDivider />
          </>
        )}
        {layerType === 'regional_guidance_layer' && (
          <MenuItem onClick={resetLayer} icon={<PiArrowCounterClockwiseBold />}>
            {t('accessibility.reset')}
          </MenuItem>
        )}
        <MenuItem onClick={deleteLayer} icon={<PiTrashSimpleBold />} color="error.300">
          {t('common.delete')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

RPLayerMenu.displayName = 'RPLayerMenu';
