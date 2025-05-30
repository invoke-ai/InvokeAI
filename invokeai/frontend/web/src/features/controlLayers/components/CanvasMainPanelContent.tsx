import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  ButtonGroup,
  ContextMenu,
  Flex,
  Heading,
  IconButton,
  Image,
  Menu,
  MenuButton,
  MenuList,
  Spacer,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
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
import { selectDynamicGrid, selectShowHUD } from 'features/controlLayers/store/canvasSettingsSlice';
import {
  canvasSessionStarted,
  selectCanvasSessionType,
  selectSelectedImage,
  selectStagedImageIndex,
  stagingAreaImageSelected,
  stagingAreaNextStagedImageSelected,
  stagingAreaPrevStagedImageSelected,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { newCanvasFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { Trans, useTranslation } from 'react-i18next';
import { PiDotsThreeOutlineVerticalFill, PiUploadBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';
import type { ProgressAndResult } from 'services/events/stores';
import { $progressImages, useMapSelector } from 'services/events/stores';
import type { Equals, Param0 } from 'tsafe';
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

const generateWithStartingImageDndTargetData = newCanvasFromImageDndTarget.getData({
  type: 'raster_layer',
  withResize: true,
});
const generateWithStartingImageAndInpaintMaskDndTargetData = newCanvasFromImageDndTarget.getData({
  type: 'raster_layer',
  withInpaintMask: true,
});
const generateWithControlImageDndTargetData = newCanvasFromImageDndTarget.getData({
  type: 'control_layer',
  withResize: true,
});

const NoActiveSession = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const newSesh = useCallback(() => {
    dispatch(canvasSessionStarted({ sessionType: 'advanced' }));
  }, [dispatch]);

  return (
    <Flex flexDir="column" w="full" h="full" alignItems="center" justifyContent="center">
      <Heading>Get Started with Invoke</Heading>
      <Button variant="ghost" onClick={newSesh}>
        Start a new Canvas Session
      </Button>
      <Text>or</Text>
      <Flex flexDir="column" maxW={512}>
        <GenerateWithStartingImage />
        <GenerateWithControlImage />
        <GenerateWithStartingImageAndInpaintMask />
      </Flex>
    </Flex>
  );
});
NoActiveSession.displayName = 'NoActiveSession';

const GenerateWithStartingImage = memo(() => {
  const { t } = useTranslation();
  const { getState, dispatch } = useAppStore();
  const useImageUploadButtonOptions = useMemo<Param0<typeof useImageUploadButton>>(
    () => ({
      onUpload: (imageDTO: ImageDTO) => {
        newCanvasFromImage({ imageDTO, type: 'raster_layer', withResize: true, getState, dispatch });
      },
      allowMultiple: false,
    }),
    [dispatch, getState]
  );
  const uploadApi = useImageUploadButton(useImageUploadButtonOptions);
  const components = useMemo(
    () => ({
      UploadButton: (
        <Button
          size="sm"
          variant="link"
          color="base.300"
          {...uploadApi.getUploadButtonProps()}
          rightIcon={<PiUploadBold />}
        />
      ),
    }),
    [uploadApi]
  );

  return (
    <Flex position="relative" flexDir="column">
      <Text fontSize="lg" fontWeight="semibold">
        Generate with a Starting Image
      </Text>
      <Text color="base.300">Regenerate the starting image using the model (Image to Image).</Text>
      <Text color="base.300">
        <Trans i18nKey="controlLayers.uploadOrDragAnImage" components={components} />
        <input {...uploadApi.getUploadInputProps()} />
      </Text>
      <DndDropTarget
        dndTarget={newCanvasFromImageDndTarget}
        dndTargetData={generateWithStartingImageDndTargetData}
        label="Drop"
      />
    </Flex>
  );
});
GenerateWithStartingImage.displayName = 'GenerateWithStartingImage';

const GenerateWithControlImage = memo(() => {
  const { t } = useTranslation();
  const { getState, dispatch } = useAppStore();
  const useImageUploadButtonOptions = useMemo<Param0<typeof useImageUploadButton>>(
    () => ({
      onUpload: (imageDTO: ImageDTO) => {
        newCanvasFromImage({ imageDTO, type: 'control_layer', withResize: true, getState, dispatch });
      },
      allowMultiple: false,
    }),
    [dispatch, getState]
  );
  const uploadApi = useImageUploadButton(useImageUploadButtonOptions);
  const components = useMemo(
    () => ({
      UploadButton: (
        <Button
          size="sm"
          variant="link"
          color="base.300"
          {...uploadApi.getUploadButtonProps()}
          rightIcon={<PiUploadBold />}
        />
      ),
    }),
    [uploadApi]
  );

  return (
    <Flex position="relative" flexDir="column">
      <Text fontSize="lg" fontWeight="semibold">
        Generate with a Control Image
      </Text>
      <Text color="base.300">
        Generate a new image using the control image to guide the structure and composition (Text to Image with
        Control).
      </Text>
      <Text color="base.300">
        <Trans i18nKey="controlLayers.uploadOrDragAnImage" components={components} />
        <input {...uploadApi.getUploadInputProps()} />
      </Text>
      <DndDropTarget
        dndTarget={newCanvasFromImageDndTarget}
        dndTargetData={generateWithControlImageDndTargetData}
        label="Drop"
      />
    </Flex>
  );
});
GenerateWithControlImage.displayName = 'GenerateWithControlImage';

const GenerateWithStartingImageAndInpaintMask = memo(() => {
  const { t } = useTranslation();
  const { getState, dispatch } = useAppStore();
  const useImageUploadButtonOptions = useMemo<Param0<typeof useImageUploadButton>>(
    () => ({
      onUpload: (imageDTO: ImageDTO) => {
        newCanvasFromImage({ imageDTO, type: 'raster_layer', withInpaintMask: true, getState, dispatch });
      },
      allowMultiple: false,
    }),
    [dispatch, getState]
  );
  const uploadApi = useImageUploadButton(useImageUploadButtonOptions);
  const components = useMemo(
    () => ({
      UploadButton: (
        <Button
          size="sm"
          variant="link"
          color="base.300"
          {...uploadApi.getUploadButtonProps()}
          rightIcon={<PiUploadBold />}
        />
      ),
    }),
    [uploadApi]
  );

  return (
    <Flex position="relative" flexDir="column">
      <Text fontSize="lg" fontWeight="semibold">
        Edit Image
      </Text>
      <Text color="base.300">Edit the image by regenerating parts of it (Inpaint).</Text>
      <Text color="base.300">
        <Trans i18nKey="controlLayers.uploadOrDragAnImage" components={components} />
        <input {...uploadApi.getUploadInputProps()} />
      </Text>
      <DndDropTarget
        dndTarget={newCanvasFromImageDndTarget}
        dndTargetData={generateWithStartingImageAndInpaintMaskDndTargetData}
        label="Drop"
      />
    </Flex>
  );
});
GenerateWithStartingImageAndInpaintMask.displayName = 'GenerateWithStartingImageAndInpaintMask';

const SimpleActiveSession = memo(() => {
  const { getState, dispatch } = useAppStore();
  const selectedImage = useAppSelector(selectSelectedImage);

  const startOver = useCallback(() => {
    dispatch(canvasSessionStarted({ sessionType: null }));
    $progressImages.set({});
  }, [dispatch]);

  const goAdvanced = useCallback(() => {
    dispatch(canvasSessionStarted({ sessionType: 'advanced' }));
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
    <Flex flexDir="column" w="full" h="full" alignItems="center" justifyContent="center" gap={2}>
      <Flex w="full">
        <Text fontSize="lg" fontWeight="bold">
          Generations from this Session
        </Text>
        <Spacer />
        <Button size="sm" onClick={startOver}>
          Start Over
        </Button>
      </Flex>
      <SelectedImageOrProgressImage />
      <SessionImages />
    </Flex>
  );
});
SimpleActiveSession.displayName = 'SimpleActiveSession';

const SelectedImageOrProgressImage = memo(() => {
  const selectedImage = useAppSelector(selectSelectedImage);

  if (selectedImage) {
    return <FullSizeImage sessionId={selectedImage.sessionId} />;
  }

  return (
    <Flex alignItems="center" justifyContent="center" minH={0} minW={0} h="full">
      <Text>No images</Text>
    </Flex>
  );
});
SelectedImageOrProgressImage.displayName = 'SelectedImageOrProgressImage';

const SelectedImage = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  const { getState, dispatch } = useAppStore();

  const vary = useCallback(() => {
    newCanvasFromImage({
      imageDTO,
      type: 'raster_layer',
      withResize: true,
      getState,
      dispatch,
    });
  }, [dispatch, getState, imageDTO]);

  const useAsControl = useCallback(() => {
    newCanvasFromImage({
      imageDTO,
      type: 'control_layer',
      withResize: true,
      getState,
      dispatch,
    });
  }, [dispatch, getState, imageDTO]);

  const edit = useCallback(() => {
    newCanvasFromImage({
      imageDTO,
      type: 'raster_layer',
      withInpaintMask: true,
      getState,
      dispatch,
    });
  }, [dispatch, getState, imageDTO]);
  return (
    <Flex position="relative" alignItems="center" justifyContent="center" minH={0} minW={0} h="full" w="full">
      <DndImage imageDTO={imageDTO} />
      <Flex position="absolute" gap={2} top={2} translateX="50%">
        <ButtonGroup isAttached={false} size="sm">
          <Button onClick={vary} tooltip="Vary the image using Image to Image">
            Vary
          </Button>
          <Button onClick={useAsControl} tooltip="Use this image to control a new Text to Image generation">
            Use as Control
          </Button>
          <Button onClick={edit} tooltip="Edit parts of this image with Inpainting">
            Edit
          </Button>
        </ButtonGroup>
      </Flex>
    </Flex>
  );
});
SelectedImage.displayName = 'SelectedImage';

const FullSizeImage = memo(({ sessionId }: { sessionId: string }) => {
  const _progressImage = useMapSelector(sessionId, $progressImages);

  if (!_progressImage) {
    return (
      <Flex alignItems="center" justifyContent="center" minH={0} minW={0} h="full">
        <Text>Pending</Text>
      </Flex>
    );
  }

  if (_progressImage.resultImage) {
    return <SelectedImage imageDTO={_progressImage.resultImage} />;
  }

  if (_progressImage.progressImage) {
    return (
      <Flex alignItems="center" justifyContent="center" minH={0} minW={0} h="full">
        <Image
          objectFit="contain"
          maxH="full"
          maxW="full"
          src={_progressImage.progressImage.dataURL}
          width={_progressImage.progressImage.width}
        />
      </Flex>
    );
  }

  return (
    <Flex alignItems="center" justifyContent="center" minH={0} minW={0} h="full">
      <Text>No progress yet</Text>
    </Flex>
  );
});
FullSizeImage.displayName = 'FullSizeImage';

const SessionImages = memo(() => {
  const progressImages = useStore($progressImages);
  return (
    <Flex position="relative" gap={2} h={108} maxW="full" overflow="scroll">
      <Spacer />
      {Object.values(progressImages).map((data, index) => {
        if (data.type === 'staged') {
          return <SessionImage key={data.sessionId} index={index} data={data} />;
        } else {
          return <ProgressImagePreview key={data.sessionId} index={index} data={data} />;
        }
      })}
      <Spacer />
    </Flex>
  );
});
SessionImages.displayName = 'SessionImages';

const ProgressImagePreview = ({ index, data }: { index: number; data: ProgressAndResult }) => {
  const dispatch = useAppDispatch();
  const selectedImageIndex = useAppSelector(selectStagedImageIndex);
  const onClick = useCallback(() => {
    dispatch(stagingAreaImageSelected({ index }));
  }, [dispatch, index]);

  useEffect(() => {
    if (selectedImageIndex === index) {
      // this doesn't work when the DndImage is in a popover... why
      document.getElementById(getStagingImageId(data.sessionId))?.scrollIntoView();
    }
  }, [data.sessionId, index, selectedImageIndex]);

  if (data.resultImage) {
    return (
      <Image
        id={getStagingImageId(data.sessionId)}
        objectFit="contain"
        maxH="full"
        maxW="full"
        src={data.resultImage.thumbnail_url}
        width={data.resultImage.width}
        onClick={onClick}
      />
    );
  }

  if (data.progressImage) {
    return (
      <Image
        id={getStagingImageId(data.sessionId)}
        objectFit="contain"
        maxH="full"
        maxW="full"
        src={data.progressImage.dataURL}
        width={data.progressImage.width}
        onClick={onClick}
      />
    );
  }

  return <Box id={getStagingImageId(data.sessionId)} bg="blue" h="full" w={108} borderWidth={1} onClick={onClick} />;
};

const getStagingImageId = (session_id: string) => `staging-image-${session_id}`;

const sx = {
  objectFit: 'contain',
  maxW: 'full',
  maxH: 'full',
  w: 'min-content',
  borderRadius: 'base',
  cursor: 'grab',
  '&[data-is-dragging=true]': {
    opacity: 0.3,
  },
  '&[data-is-selected="false"]': {
    opacity: 0.5,
  },
} satisfies SystemStyleObject;
const SessionImage = memo(({ index, data }: { index: number; data: ProgressAndResult }) => {
  const dispatch = useAppDispatch();
  const selectedImageIndex = useAppSelector(selectStagedImageIndex);
  const onClick = useCallback(() => {
    dispatch(stagingAreaImageSelected({ index }));
  }, [dispatch, index]);
  useEffect(() => {
    if (selectedImageIndex === index) {
      // this doesn't work when the DndImage is in a popover... why
      document.getElementById(getStagingImageId(data.sessionId))?.scrollIntoView();
    }
  }, [data.sessionId, index, selectedImageIndex]);
  return (
    <DndImage
      id={getStagingImageId(data.sessionId)}
      imageDTO={data.imageDTO}
      asThumbnail
      onClick={onClick}
      data-is-selected={selectedImageIndex === index}
      w={data.imageDTO.width}
      sx={sx}
      borderWidth={1}
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
