import { ContextMenu, Flex, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion } from 'common/hooks/focus';
import { CanvasAlertsPreserveMask } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsPreserveMask';
import { CanvasAlertsSelectedEntityStatus } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSelectedEntityStatus';
import { CanvasAlertsSendingToGallery } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSendingTo';
import { CanvasContextMenuGlobalMenuItems } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuGlobalMenuItems';
import { CanvasContextMenuSelectedEntityMenuItems } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuSelectedEntityMenuItems';
import { CanvasDropArea } from 'features/controlLayers/components/CanvasDropArea';
import { Filter } from 'features/controlLayers/components/Filters/Filter';
import { CanvasHUD } from 'features/controlLayers/components/HUD/CanvasHUD';
import { InvokeCanvasComponent } from 'features/controlLayers/components/InvokeCanvasComponent';
import { StagingAreaIsStagingGate } from 'features/controlLayers/components/StagingArea/StagingAreaIsStagingGate';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { CanvasToolbar } from 'features/controlLayers/components/Toolbar/CanvasToolbar';
import { Transform } from 'features/controlLayers/components/Transform/Transform';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectDynamicGrid, selectShowHUD } from 'features/controlLayers/store/canvasSettingsSlice';
import { GatedImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import { memo, useCallback, useRef } from 'react';
import { PiDotsThreeOutlineVerticalFill } from 'react-icons/pi';

const MenuContent = () => {
  return (
    <CanvasManagerProviderGate>
      <MenuList>
        <CanvasContextMenuGlobalMenuItems />
        <CanvasContextMenuSelectedEntityMenuItems />
      </MenuList>
    </CanvasManagerProviderGate>
  );
};

export const CanvasMainPanelContent = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const dynamicGrid = useAppSelector(selectDynamicGrid);
  const showHUD = useAppSelector(selectShowHUD);

  const renderMenu = useCallback(() => {
    return <MenuContent />;
  }, []);

  useFocusRegion('canvas', ref);

  return (
    <Flex
      tabIndex={-1}
      ref={ref}
      borderRadius="base"
      position="relative"
      flexDirection="column"
      height="full"
      width="full"
      gap={2}
      alignItems="center"
      justifyContent="center"
    >
      <CanvasManagerProviderGate>
        <CanvasToolbar />
      </CanvasManagerProviderGate>
      <ContextMenu<HTMLDivElement> renderMenu={renderMenu} withLongPress={false}>
        {(ref) => (
          <Flex
            ref={ref}
            position="relative"
            w="full"
            h="full"
            bg={dynamicGrid ? 'base.850' : 'base.900'}
            borderRadius="base"
          >
            <InvokeCanvasComponent />
            <CanvasManagerProviderGate>
              {showHUD && (
                <Flex position="absolute" top={1} insetInlineStart={1} pointerEvents="none">
                  <CanvasHUD />
                </Flex>
              )}
              <Flex flexDir="column" position="absolute" top={1} insetInlineEnd={1} pointerEvents="none" gap={2}>
                <CanvasAlertsSelectedEntityStatus />
                <CanvasAlertsPreserveMask />
                <CanvasAlertsSendingToGallery />
              </Flex>
              <Flex position="absolute" top={1} insetInlineEnd={1}>
                <Menu>
                  <MenuButton as={IconButton} icon={<PiDotsThreeOutlineVerticalFill />} colorScheme="base" />
                  <MenuContent />
                </Menu>
              </Flex>
            </CanvasManagerProviderGate>
          </Flex>
        )}
      </ContextMenu>
      <Flex position="absolute" bottom={4} gap={2} align="center" justify="center">
        <CanvasManagerProviderGate>
          <StagingAreaIsStagingGate>
            <StagingAreaToolbar />
          </StagingAreaIsStagingGate>
        </CanvasManagerProviderGate>
      </Flex>
      <Flex position="absolute" bottom={4}>
        <CanvasManagerProviderGate>
          <Filter />
          <Transform />
        </CanvasManagerProviderGate>
      </Flex>
      <CanvasDropArea />
      <GatedImageViewer />
    </Flex>
  );
});

CanvasMainPanelContent.displayName = 'CanvasMainPanelContent';
