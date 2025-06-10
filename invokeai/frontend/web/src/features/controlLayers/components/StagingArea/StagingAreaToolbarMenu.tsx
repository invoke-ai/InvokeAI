import { IconButton, Menu, MenuButton, MenuDivider, MenuList } from '@invoke-ai/ui-library';
import { StagingAreaToolbarMenuAutoSwitch } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarMenuAutoSwitch';
import { StagingAreaToolbarNewLayerFromImageMenuItems } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarMenuNewLayerFromImage';
import { memo } from 'react';
import { PiDotsThreeBold } from 'react-icons/pi';

export const StagingAreaToolbarMenu = memo(() => {
  return (
    <Menu>
      <MenuButton as={IconButton} icon={<PiDotsThreeBold />} colorScheme="invokeBlue" />
      <MenuList>
        <StagingAreaToolbarMenuAutoSwitch />
        <MenuDivider />
        <StagingAreaToolbarNewLayerFromImageMenuItems />
      </MenuList>
    </Menu>
  );
});

StagingAreaToolbarMenu.displayName = 'StagingAreaToolbarMenu';
