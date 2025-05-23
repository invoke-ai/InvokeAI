import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Button, ContextMenu, Flex, IconButton, Image, Menu, MenuButton, MenuList, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { CanvasAlertsPreserveMask } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsPreserveMask';
import { CanvasAlertsSelectedEntityStatus } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSelectedEntityStatus';
import { CanvasAlertsSendingToGallery } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSendingTo';
import { CanvasContextMenuGlobalMenuItems } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuGlobalMenuItems';
import { CanvasContextMenuSelectedEntityMenuItems } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuSelectedEntityMenuItems';
import { CanvasDropArea } from 'features/controlLayers/components/CanvasDropArea';
import { Filter } from 'features/controlLayers/components/Filters/Filter';
import { CanvasHUD } from 'features/controlLayers/components/HUD/CanvasHUD';
import { InvokeCanvasComponent } from 'features/controlLayers/components/InvokeCanvasComponent';
import { SelectObject } from 'features/controlLayers/components/SelectObject/SelectObject';
import { StagingAreaIsStagingGate } from 'features/controlLayers/components/StagingArea/StagingAreaIsStagingGate';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { CanvasToolbar } from 'features/controlLayers/components/Toolbar/CanvasToolbar';
import { Transform } from 'features/controlLayers/components/Transform/Transform';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { newCanvasSessionRequested } from 'features/controlLayers/store/actions';
import { selectDynamicGrid, selectShowHUD } from 'features/controlLayers/store/canvasSettingsSlice';
import {
  selectIsStaging,
  selectSelectedImage,
  selectStagedImages,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectIsCanvasEmpty, selectIsSessionStarted } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { PiDotsThreeOutlineVerticalFill } from 'react-icons/pi';
import { assert } from 'tsafe';

import { CanvasAlertsInvocationProgress } from './CanvasAlerts/CanvasAlertsInvocationProgress';

const FOCUS_REGION_STYLES: SystemStyleObject = {
  width: 'full',
  height: 'full',
};

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

export const CanvasMainPanelContent = memo(() => {
  const isCanvasEmpty = useAppSelector(selectIsCanvasEmpty);
  const isSessionStarted = useAppSelector(selectIsSessionStarted);

  if (!isSessionStarted) {
    return <NoActiveSession />;
  }

  if (isSessionStarted && isCanvasEmpty) {
    return <SimpleActiveSession />;
  }

  if (isSessionStarted && !isCanvasEmpty) {
    return <CanvasActiveSession />;
  }

  assert(false);
});

CanvasMainPanelContent.displayName = 'CanvasMainPanelContent';

const NoActiveSession = memo(() => {
  const dispatch = useAppDispatch();
  const newSesh = useCallback(() => {
    dispatch(newCanvasSessionRequested());
  }, [dispatch]);
  return (
    <Flex flexDir="column" w="full" h="full" alignItems="center" justifyContent="center">
      <Text fontSize="lg" fontWeight="bold">
        No Active Session
      </Text>
      <Button display="flex" flexDir="column" gap={2} p={8} minH={0} minW={0} onClick={newSesh}>
        <Text>New Canvas Session</Text>
        <Text>- New Canvas Session</Text>
        <Text>- 1 Inpaint mask layer added</Text>
      </Button>
      <Flex flexDir="column" gap={2} p={8} border="dashed yellow 2px">
        <Text>Generate with Starting Image</Text>
        <Text>- New Canvas Session</Text>
        <Text>- Dropped image as raster layer</Text>
        <Text>- Bbox resized</Text>
      </Flex>
      <Flex flexDir="column" gap={2} p={8} border="dashed yellow 2px">
        <Text>Generate with Control Image</Text>
        <Text>- New Canvas Session</Text>
        <Text>- Dropped image as control layer</Text>
        <Text>- Bbox resized</Text>
      </Flex>
      <Flex flexDir="column" gap={2} p={8} border="dashed yellow 2px">
        <Text>Edit Image</Text>
        <Text>- New Canvas Session</Text>
        <Text>- Dropped image as raster layer</Text>
        <Text>- Bbox resized</Text>
        <Text>- 1 Inpaint mask layer added</Text>
      </Flex>
    </Flex>
  );
});
NoActiveSession.displayName = 'NoActiveSession';
const SimpleActiveSession = memo(() => {
  const isStaging = useAppSelector(selectIsStaging);
  const selectedImage = useAppSelector(selectSelectedImage);
  const stagedImages = useAppSelector(selectStagedImages);
  return (
    <Flex flexDir="column" w="full" h="full" alignItems="center" justifyContent="center">
      <Text fontSize="lg" fontWeight="bold">
        Simple Session (staging view) {isStaging && 'STAGING'}
      </Text>
      {selectedImage && <Image src={selectedImage.imageDTO.image_url} />}
      <Flex gap={2} maxW="full" overflow="scroll">
        {stagedImages.map(({ imageDTO }) => (
          <Image key={imageDTO.image_name} maxW={108} src={imageDTO.thumbnail_url} />
        ))}
      </Flex>
    </Flex>
  );
});
SimpleActiveSession.displayName = 'SimpleActiveSession';

const CanvasActiveSession = memo(() => {
  const dynamicGrid = useAppSelector(selectDynamicGrid);
  const showHUD = useAppSelector(selectShowHUD);

  const renderMenu = useCallback(() => {
    return <MenuContent />;
  }, []);

  return (
    <FocusRegionWrapper region="canvas" sx={FOCUS_REGION_STYLES}>
      <Flex
        tabIndex={-1}
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
        <ContextMenu<HTMLDivElement> renderMenu={renderMenu} withLongPress={false}>
          {(ref) => (
            <Flex
              ref={ref}
              position="relative"
              w="full"
              h="full"
              bg={dynamicGrid ? 'base.850' : 'base.900'}
              borderRadius="base"
              overflow="hidden"
            >
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
                  <CanvasAlertsSendingToGallery />
                  <CanvasAlertsInvocationProgress />
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
            <SelectObject />
          </CanvasManagerProviderGate>
        </Flex>
        <CanvasManagerProviderGate>
          <CanvasDropArea />
        </CanvasManagerProviderGate>
      </Flex>
    </FocusRegionWrapper>
  );
});
CanvasActiveSession.displayName = 'ActiveCanvasContent';
