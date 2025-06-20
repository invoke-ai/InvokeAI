import { IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { StagingAreaToolbarMenuAutoSwitch } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarMenuAutoSwitch';
import { memo } from 'react';
import { PiDotsThreeBold } from 'react-icons/pi';

export const SimpleStagingAreaToolbarMenu = memo(() => {
  return (
    <Menu>
      <MenuButton as={IconButton} icon={<PiDotsThreeBold />} colorScheme="invokeBlue" />
      <MenuList>
        <StagingAreaToolbarMenuAutoSwitch />
      </MenuList>
    </Menu>
  );
});

SimpleStagingAreaToolbarMenu.displayName = 'SimpleStagingAreaToolbarMenu';
