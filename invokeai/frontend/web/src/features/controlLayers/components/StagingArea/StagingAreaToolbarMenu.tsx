import { IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { StagingAreaToolbarNewLayerFromImageMenuItems } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarMenuNewLayerFromImage';
import { useCanvasManager } from 'features/controlLayers/hooks/useCanvasManager';
import { memo } from 'react';
import { PiDotsThreeVerticalBold } from 'react-icons/pi';

export const StagingAreaToolbarMenu = memo(() => {
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  return (
    <Menu>
      <MenuButton
        tooltip="Image Actions"
        as={IconButton}
        icon={<PiDotsThreeVerticalBold />}
        colorScheme="invokeBlue"
        isDisabled={!shouldShowStagedImage}
      />
      <MenuList>
        <StagingAreaToolbarNewLayerFromImageMenuItems />
      </MenuList>
    </Menu>
  );
});

StagingAreaToolbarMenu.displayName = 'StagingAreaToolbarMenu';
