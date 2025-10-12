import { ContextMenu, Divider, Flex, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAlertsBboxVisibility } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsBboxVisibility';
import { CanvasAlertsInvocationProgress } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsInvocationProgress';
import { CanvasAlertsPreserveMask } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsPreserveMask';
import { CanvasAlertsSaveAllImagesToGallery } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSaveAllImagesToGallery';
import { CanvasAlertsSelectedEntityStatus } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSelectedEntityStatus';
import { CanvasBusySpinner } from 'features/controlLayers/components/CanvasBusySpinner';
import { CanvasContextMenuGlobalMenuItems } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuGlobalMenuItems';
import { CanvasContextMenuSelectedEntityMenuItems } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuSelectedEntityMenuItems';
import { CanvasDropArea } from 'features/controlLayers/components/CanvasDropArea';
import { Filter } from 'features/controlLayers/components/Filters/Filter';
import { CanvasHUD } from 'features/controlLayers/components/HUD/CanvasHUD';
import { InvokeCanvasComponent } from 'features/controlLayers/components/InvokeCanvasComponent';
import { SelectObject } from 'features/controlLayers/components/SelectObject/SelectObject';
import { StagingAreaContextProvider } from 'features/controlLayers/components/StagingArea/context';
import { CanvasToolbar } from 'features/controlLayers/components/Toolbar/CanvasToolbar';
import { Transform } from 'features/controlLayers/components/Transform/Transform';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectDynamicGrid, selectShowHUD } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { PiDotsThreeOutlineVerticalFill } from 'react-icons/pi';

import { CanvasTabs } from './CanvasTabs';
import { StagingArea } from './StagingArea';

const MenuContent = memo(() => {
  return (
    <CanvasManagerProviderGate>
      <MenuList>
        <CanvasContextMenuSelectedEntityMenuItems />
        <CanvasContextMenuGlobalMenuItems />
      </MenuList>
    </CanvasManagerProviderGate>
  );
});
MenuContent.displayName = 'MenuContent';

const canvasBgSx = {
  position: 'relative',
  w: 'full',
  h: 'full',
  borderRadius: 'base',
  overflow: 'hidden',
  bg: 'base.900',
  '&[data-dynamic-grid="true"]': {
    bg: 'base.850',
  },
};

const ActiveCanvas = memo(() => {
  const dynamicGrid = useAppSelector((state) => selectDynamicGrid(state));
  const showHUD = useAppSelector((state) => selectShowHUD(state));

  const renderMenu = useCallback(() => {
    return <MenuContent />;
  }, []);

  return (
    <Flex w="full" h="full">
      <StagingAreaContextProvider>
        <ContextMenu<HTMLDivElement> renderMenu={renderMenu} withLongPress={false}>
          {(ref) => (
            <Flex ref={ref} sx={canvasBgSx} data-dynamic-grid={dynamicGrid}>
              <InvokeCanvasComponent />
              <CanvasManagerProviderGate>
                <Flex
                  position="absolute"
                  flexDir="column"
                  top={1}
                  insetInlineStart={1}
                  pointerEvents="none"
                  gap={2}
                  alignItems="flex-start"
                >
                  {showHUD && <CanvasHUD />}
                  <CanvasAlertsSaveAllImagesToGallery />
                  <CanvasAlertsSelectedEntityStatus />
                  <CanvasAlertsPreserveMask />
                  <CanvasAlertsInvocationProgress />
                  <CanvasAlertsBboxVisibility />
                </Flex>
                <Flex position="absolute" top={1} insetInlineEnd={1}>
                  <Menu>
                    <MenuButton as={IconButton} icon={<PiDotsThreeOutlineVerticalFill />} colorScheme="base" />
                    <MenuContent />
                  </Menu>
                </Flex>
                <CanvasBusySpinner position="absolute" insetInlineEnd={2} bottom={2} />
              </CanvasManagerProviderGate>
            </Flex>
          )}
        </ContextMenu>
        <CanvasManagerProviderGate>
          <StagingArea />
        </CanvasManagerProviderGate>
        <Flex position="absolute" bottom={4}>
          <CanvasManagerProviderGate>
            <Filter />
            <Transform />
            <SelectObject />
          </CanvasManagerProviderGate>
        </Flex>
        <CanvasManagerProviderGate>
          <CanvasDropArea />
        </CanvasManagerProviderGate>
      </StagingAreaContextProvider>
    </Flex>
  );
});
ActiveCanvas.displayName = 'ActiveCanvas';

export const CanvasWorkspacePanel = memo(() => {
  return (
    <Flex
      borderRadius="base"
      position="relative"
      flexDirection="column"
      height="full"
      width="full"
      gap={2}
      alignItems="center"
      justifyContent="center"
      overflow="hidden"
    >
      <CanvasManagerProviderGate>
        <CanvasToolbar />
      </CanvasManagerProviderGate>
      <Divider />
      <CanvasTabs />
      <ActiveCanvas />
    </Flex>
  );
});
CanvasWorkspacePanel.displayName = 'CanvasWorkspacePanel';
