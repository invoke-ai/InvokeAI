import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Button, ContextMenu, Flex, IconButton, Image, Menu, MenuButton, MenuList, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
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
import { canvasReset, newAdvancedCanvasSessionRequested } from 'features/controlLayers/store/actions';
import { selectDynamicGrid, selectShowHUD } from 'features/controlLayers/store/canvasSettingsSlice';
import {
  selectCanvasSessionType,
  selectIsStaging,
  selectSelectedImage,
  selectStagedImageIndex,
  selectStagedImages,
  stagingAreaImageSelected,
  stagingAreaImageStaged,
  stagingAreaNextStagedImageSelected,
  stagingAreaPrevStagedImageSelected,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { isImageField, type ProgressImage } from 'features/nodes/types/common';
import { isCanvasOutputEvent } from 'features/nodes/util/graph/graphBuilderUtils';
import type { Atom } from 'nanostores';
import { atom } from 'nanostores';
import { memo, useCallback, useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { useHotkeys } from 'react-hotkeys-hook';
import { PiDotsThreeOutlineVerticalFill } from 'react-icons/pi';
import { getImageDTOSafe } from 'services/api/endpoints/images';
import type { ImageDTO, S } from 'services/api/types';
import { $socket } from 'services/events/stores';
import type { Equals } from 'tsafe';
import { assert, objectEntries } from 'tsafe';

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
  const sessionType = useAppSelector(selectCanvasSessionType);

  if (sessionType === null) {
    return <NoActiveSession />;
  }

  if (sessionType === 'simple') {
    return <SimpleActiveSession />;
  }

  if (sessionType === 'advanced') {
    return <CanvasActiveSession />;
  }

  assert<Equals<never, typeof sessionType>>(false, 'Unexpected sessionType');
});

CanvasMainPanelContent.displayName = 'CanvasMainPanelContent';

const NoActiveSession = memo(() => {
  const dispatch = useAppDispatch();
  const newSesh = useCallback(() => {
    dispatch(newAdvancedCanvasSessionRequested());
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

type EphemeralProgressImage = { sessionId: string; image: ProgressImage };

const SimpleActiveSession = memo(() => {
  const dispatch = useAppDispatch();
  const isStaging = useAppSelector(selectIsStaging);
  const socket = useStore($socket);
  const [$progressImage] = useState(() => atom<EphemeralProgressImage | null>(null));

  useEffect(() => {
    if (!socket) {
      return;
    }
    const onInvocationProgress = (event: S['InvocationProgressEvent']) => {
      if (!event) {
        return;
      }
      if (event.origin !== 'canvas') {
        return;
      }
      if (!event.image) {
        return;
      }
      $progressImage.set({ sessionId: event.session_id, image: event.image });
    };
    const onInvocationComplete = async (event: S['InvocationCompleteEvent']) => {
      const progressImage = $progressImage.get();
      if (!progressImage) {
        return;
      }
      if (progressImage.sessionId !== event.session_id) {
        return;
      }
      if (!isCanvasOutputEvent(event)) {
        return;
      }
      let imageDTO: ImageDTO | null = null;
      for (const [_name, value] of objectEntries(event.result)) {
        if (isImageField(value)) {
          imageDTO = await getImageDTOSafe(value.image_name);
          break;
        }
      }
      if (!imageDTO) {
        return;
      }
      flushSync(() => {
        dispatch(stagingAreaImageStaged({ stagingAreaImage: { imageDTO, offsetX: 0, offsetY: 0 } }));
      });
      $progressImage.set(null);
    };

    const onQueueItemStatusChanged = (event: S['QueueItemStatusChangedEvent']) => {
      const progressImage = $progressImage.get();
      if (!progressImage) {
        return;
      }
      if (progressImage.sessionId !== event.session_id) {
        return;
      }
      if (event.status !== 'canceled' && event.status !== 'failed') {
        return;
      }
      $progressImage.set(null);
    };
    console.log('SUB session preview image listeners');
    socket.on('invocation_progress', onInvocationProgress);
    socket.on('invocation_complete', onInvocationComplete);
    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      console.log('UNSUB session preview image listeners');
      socket.off('invocation_progress', onInvocationProgress);
      socket.off('invocation_complete', onInvocationComplete);
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [$progressImage, dispatch, socket]);

  const onReset = useCallback(() => {
    dispatch(canvasReset());
  }, [dispatch]);

  const selectNext = useCallback(() => {
    dispatch(stagingAreaNextStagedImageSelected());
  }, [dispatch]);

  useHotkeys(['right'], selectNext, { preventDefault: true }, [selectNext]);

  const selectPrev = useCallback(() => {
    dispatch(stagingAreaPrevStagedImageSelected());
  }, [dispatch]);

  useHotkeys(['left'], selectPrev, { preventDefault: true }, [selectPrev]);

  return (
    <Flex flexDir="column" w="full" h="full" alignItems="center" justifyContent="center">
      <Flex>
        <Text fontSize="lg" fontWeight="bold">
          Simple Session (staging view) {isStaging && 'STAGING'}
        </Text>
        <Button onClick={onReset}>reset</Button>
      </Flex>
      <SelectedImage $progressImage={$progressImage} />
      <SessionImages />
    </Flex>
  );
});
SimpleActiveSession.displayName = 'SimpleActiveSession';

const SelectedImage = memo(({ $progressImage }: { $progressImage: Atom<EphemeralProgressImage | null> }) => {
  const progressImage = useStore($progressImage);
  const selectedImage = useAppSelector(selectSelectedImage);

  if (progressImage) {
    return (
      <Flex alignItems="center" justifyContent="center" minH={0} minW={0}>
        <Image
          objectFit="contain"
          maxH="full"
          maxW="full"
          src={progressImage.image.dataURL}
          width={progressImage.image.width}
          height={progressImage.image.height}
        />
      </Flex>
    );
  }

  if (selectedImage) {
    return (
      <Flex alignItems="center" justifyContent="center" minH={0} minW={0}>
        <Image
          objectFit="contain"
          maxH="full"
          maxW="full"
          src={selectedImage.imageDTO.image_url}
          width={selectedImage.imageDTO.width}
          height={selectedImage.imageDTO.height}
        />
      </Flex>
    );
  }

  return <Text>No images</Text>;
});
SelectedImage.displayName = 'SelectedImage';

const SessionImages = memo(() => {
  const stagedImages = useAppSelector(selectStagedImages);
  return (
    <Flex gap={2} h={108} maxW="full" overflow="scroll">
      {stagedImages.map(({ imageDTO }, index) => (
        <SessionImage key={imageDTO.image_name} index={index} imageDTO={imageDTO} />
      ))}
    </Flex>
  );
});
SessionImages.displayName = 'SessionImages';

const sx = {
  '&[data-is-selected="false"]': {
    opacity: 0.5,
  },
} satisfies SystemStyleObject;
const SessionImage = memo(({ index, imageDTO }: { index: number; imageDTO: ImageDTO }) => {
  const dispatch = useAppDispatch();
  const selectedImageIndex = useAppSelector(selectStagedImageIndex);
  const onClick = useCallback(() => {
    dispatch(stagingAreaImageSelected({ index }));
  }, [dispatch, index]);
  return (
    <Image
      maxW={108}
      src={imageDTO.image_url}
      fallbackSrc={imageDTO.thumbnail_url}
      onClick={onClick}
      data-is-selected={selectedImageIndex === index}
      sx={sx}
    />
  );
});
SessionImage.displayName = 'SessionImage';

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
