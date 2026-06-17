import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityConvertTo3D } from 'features/controlLayers/hooks/useEntityConvertTo3D';
import { memo, useCallback } from 'react';
import { PiCubeFill } from 'react-icons/pi';

export const CanvasEntityMenuItemsConvertTo3D = memo(() => {
  const subMenu = useSubMenu();
  const entityIdentifier = useEntityIdentifierContext();
  const { isDisabled, start } = useEntityConvertTo3D(entityIdentifier);
  const convertIsolated = useCallback(() => start(true), [start]);
  const convertFullImage = useCallback(() => start(false), [start]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiCubeFill />} isDisabled={isDisabled}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label="Convert to 3D" />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem onClick={convertIsolated} icon={<PiCubeFill />} isDisabled={isDisabled}>
            Isolate subject
          </MenuItem>
          <MenuItem onClick={convertFullImage} icon={<PiCubeFill />} isDisabled={isDisabled}>
            Keep background
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

CanvasEntityMenuItemsConvertTo3D.displayName = 'CanvasEntityMenuItemsConvertTo3D';
