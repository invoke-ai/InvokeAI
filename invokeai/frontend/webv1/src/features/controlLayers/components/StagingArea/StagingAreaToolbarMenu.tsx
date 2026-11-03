import { IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { StagingAreaToolbarNewLayerFromImageMenuItems } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarMenuNewLayerFromImage';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotsThreeVerticalBold } from 'react-icons/pi';

export const StagingAreaToolbarMenu = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  return (
    <Menu>
      <MenuButton
        tooltip={t('parameters.imageActions')}
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
