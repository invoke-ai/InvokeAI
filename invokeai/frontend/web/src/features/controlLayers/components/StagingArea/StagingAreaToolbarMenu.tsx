import { IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { StagingAreaToolbarNewLayerFromImageMenuItems } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarMenuNewLayerFromImage';
import { memo } from 'react';
import { PiDotsThreeVerticalBold } from 'react-icons/pi';

export const StagingAreaToolbarMenu = memo(() => {
  return (
    <Menu>
      <MenuButton as={IconButton} icon={<PiDotsThreeVerticalBold />} colorScheme="invokeBlue" />
      <MenuList>
        <StagingAreaToolbarNewLayerFromImageMenuItems />
      </MenuList>
    </Menu>
  );
});

StagingAreaToolbarMenu.displayName = 'StagingAreaToolbarMenu';
