import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityConvertTo3D } from 'features/controlLayers/hooks/useEntityConvertTo3D';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCubeFill } from 'react-icons/pi';

export const CanvasEntityMenuItemsConvertTo3D = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const entityIdentifier = useEntityIdentifierContext();
  const { isDisabled, start } = useEntityConvertTo3D(entityIdentifier);
  const convertIsolated = useCallback(() => start(true), [start]);
  const convertFullImage = useCallback(() => start(false), [start]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiCubeFill />} isDisabled={isDisabled}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.convertTo3D.convertTo3D')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem onClick={convertIsolated} icon={<PiCubeFill />} isDisabled={isDisabled}>
            {t('controlLayers.convertTo3D.isolateSubject')}
          </MenuItem>
          <MenuItem onClick={convertFullImage} icon={<PiCubeFill />} isDisabled={isDisabled}>
            {t('controlLayers.convertTo3D.keepBackground')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

CanvasEntityMenuItemsConvertTo3D.displayName = 'CanvasEntityMenuItemsConvertTo3D';
