import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { PBRProcessingRequested } from 'app/store/middleware/listenerMiddleware/listeners/addPBRFilterListener';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useCanvasIsBusySafe } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileBold, PiShootingStarFill } from 'react-icons/pi';

export const ContextMenuItemFiltersSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();

  const isBusy = useCanvasIsBusySafe();
  const isStaging = useCanvasIsStaging();

  const handleClickPBRMaps = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(PBRProcessingRequested({ imageDTO }));
  }, [dispatch, imageDTO]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiShootingStarFill />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.filter.filters')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem icon={<PiFileBold />} isDisabled={isStaging || isBusy} onClick={handleClickPBRMaps}>
            {t('controlLayers.filter.pbr_maps.label')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ContextMenuItemFiltersSubMenu.displayName = 'ContextMenuItemFiltersSubMenu';
