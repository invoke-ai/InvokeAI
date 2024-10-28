import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { CanvasEntityMenuItemsCopyToClipboard } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCopyToClipboard';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import { inpaintMaskConvertedToRegionalGuidance } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

export const InpaintMaskMenuItemsCopyToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const isInteractable = useIsEntityInteractable(entityIdentifier);

  const copyToRegionalGuidance = useCallback(() => {
    dispatch(inpaintMaskConvertedToRegionalGuidance({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiCopyBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.copyInpaintMaskTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <CanvasEntityMenuItemsCopyToClipboard />
          <MenuItem onClick={copyToRegionalGuidance} icon={<PiCopyBold />} isDisabled={!isInteractable}>
            {t('controlLayers.newRegionalGuidance')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

InpaintMaskMenuItemsCopyToSubMenu.displayName = 'InpaintMaskMenuItemsCopyToSubMenu';
