import { ContextMenu, Divider, Flex, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
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
import { CanvasSessionContextProvider } from 'features/controlLayers/components/SimpleSession/context';
import { StagingAreaItemsList } from 'features/controlLayers/components/SimpleSession/StagingAreaItemsList';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { CanvasToolbar } from 'features/controlLayers/components/Toolbar/CanvasToolbar';
import { Transform } from 'features/controlLayers/components/Transform/Transform';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectDynamicGrid, selectShowHUD } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectCanvasSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { PiDotsThreeOutlineVerticalFill } from 'react-icons/pi';

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

export const CanvasWorkspacePanel = memo(() => {
  const dynamicGrid = useAppSelector(selectDynamicGrid);
  const showHUD = useAppSelector(selectShowHUD);
  const canvasId = useAppSelector(selectCanvasSessionId);

  const renderMenu = useCallback(() => {
    return <MenuContent />;
  }, []);

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
                <CanvasAlertsSelectedEntityStatus />
                <CanvasAlertsPreserveMask />
                <CanvasAlertsSaveAllImagesToGallery />
                <CanvasAlertsInvocationProgress />
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
      {canvasId !== null && (
        <CanvasManagerProviderGate>
          <CanvasSessionContextProvider type="advanced" id={canvasId}>
            <Flex
              position="absolute"
              flexDir="column"
              bottom={4}
              gap={2}
              align="center"
              justify="center"
              left={4}
              right={4}
            >
              <Flex position="relative" maxW="full" w="full" h={108}>
                <StagingAreaItemsList />
              </Flex>
              <Flex gap={2}>
                <StagingAreaToolbar />
              </Flex>
            </Flex>
          </CanvasSessionContextProvider>
        </CanvasManagerProviderGate>
      )}
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
    </Flex>
  );
});
CanvasWorkspacePanel.displayName = 'CanvasWorkspacePanel';
