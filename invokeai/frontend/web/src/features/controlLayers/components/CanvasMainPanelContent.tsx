/* eslint-disable i18next/no-literal-string */
import type { FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  ButtonGroup,
  CircularProgress,
  ContextMenu,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  IconButton,
  Image,
  Menu,
  MenuButton,
  MenuList,
  Spacer,
  Switch,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { CanvasAlertsPreserveMask } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsPreserveMask';
import { CanvasAlertsSelectedEntityStatus } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSelectedEntityStatus';
import { CanvasAlertsSendingToGallery } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSendingTo';
import { CanvasBusySpinner } from 'features/controlLayers/components/CanvasBusySpinner';
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
import type { ProgressImage } from 'features/nodes/types/common';
import { isImageField } from 'features/nodes/types/common';
import { isCanvasOutputNodeId } from 'features/nodes/util/graph/graphBuilderUtils';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { Trans, useTranslation } from 'react-i18next';
import { PiDotsThreeOutlineVerticalFill, PiUploadBold } from 'react-icons/pi';
import { getImageDTOSafe, useGetImageDTOQuery } from 'services/api/endpoints/images';
import { useListAllQueueItemsQuery } from 'services/api/endpoints/queue';
import type { ImageDTO, S } from 'services/api/types';
import type { ProgressAndResult } from 'services/events/stores';
import { $progressImages, $socket, useMapSelector } from 'services/events/stores';
import type { Equals, Param0 } from 'tsafe';
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
          Generations
        </Text>
        <Spacer />
        <Button size="sm" variant="ghost" onClick={startOver}>
          Start Over
        </Button>
      </Flex>
      <StagingArea />
    </Flex>
  );
});
SimpleActiveSession.displayName = 'SimpleActiveSession';

const scrollIndicatorSx = {
  opacity: 0,
  '&[data-visible="true"]': {
    opacity: 1,
  },
} satisfies SystemStyleObject;

const StagingArea = memo(() => {
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null);
  const [autoSwitch, setAutoSwitch] = useState(true);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const scrollableRef = useRef<HTMLDivElement>(null);
  const { data } = useListAllQueueItemsQuery({ destination: 'canvas' });
  const items = useMemo(() => data?.filter(({ status }) => status !== 'canceled') ?? EMPTY_ARRAY, [data]);
  const selectedItem = useMemo(
    () =>
      items.length > 0 && selectedItemId !== null ? items.find(({ item_id }) => item_id === selectedItemId) : null,
    [items, selectedItemId]
  );

  useEffect(() => {
    if (items.length === 0) {
      setSelectedItemId(null);
      return;
    }
    if (selectedItem === null && items.length > 0) {
      setSelectedItemId(items[0]?.item_id ?? null);
      return;
    }
    if (selectedItemId === null || items.find((item) => item.item_id === selectedItemId) === undefined) {
      return;
    }
    document.getElementById(`queue-item-status-card-${selectedItemId}`)?.scrollIntoView();
  }, [items, selectedItem, selectedItemId]);

  useEffect(() => {
    const el = scrollableRef.current;
    if (!el) {
      return;
    }
    const onScroll = () => {
      const { scrollLeft, scrollWidth, clientWidth } = el;
      setCanScrollLeft(scrollLeft > 0);
      setCanScrollRight(scrollLeft + clientWidth < scrollWidth);
    };
    el.addEventListener('scroll', onScroll);
    const observer = new ResizeObserver(onScroll);
    observer.observe(el);
    return () => {
      el.removeEventListener('scroll', onScroll);
      observer.disconnect();
    };
  }, []);

  const onSelectItem = useCallback((item: S['SessionQueueItem']) => {
    setSelectedItemId(item.item_id);
    if (item.status !== 'in_progress') {
      setAutoSwitch(false);
    }
  }, []);

  const onNext = useCallback(() => {
    if (selectedItemId === null) {
      return;
    }
    const currentIndex = items.findIndex((item) => item.item_id === selectedItemId);
    const nextIndex = (currentIndex + 1) % items.length;
    const nextItem = items[nextIndex];
    if (!nextItem) {
      return;
    }
    setSelectedItemId(nextItem.item_id);
  }, [items, selectedItemId]);
  const onPrev = useCallback(() => {
    if (selectedItemId === null) {
      return;
    }
    const currentIndex = items.findIndex((item) => item.item_id === selectedItemId);
    const prevIndex = (currentIndex - 1 + items.length) % items.length;
    const prevItem = items[prevIndex];
    if (!prevItem) {
      return;
    }
    setSelectedItemId(prevItem.item_id);
  }, [items, selectedItemId]);

  useHotkeys('left', onPrev);
  useHotkeys('right', onNext);

  const socket = useStore($socket);
  useEffect(() => {
    if (!autoSwitch) {
      return;
    }

    if (!socket) {
      return;
    }

    const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
      if (data.destination !== 'canvas') {
        return;
      }
      if (data.status === 'in_progress') {
        setSelectedItemId(data.item_id);
      }
    };

    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [autoSwitch, socket]);

  const onChangeAutoSwitch = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setAutoSwitch(e.target.checked);
  }, []);

  return (
    <Flex position="relative" flexDir="column" gap={2} w="full" h="full" minW={0} minH={0}>
      <Flex w="full" h="full" alignItems="center" justifyContent="center" minW={0} minH={0}>
        {selectedItem && <QueueItemStatusCard item={selectedItem} minW={0} minH={0} h="full" isSelected={false} />}
        {!selectedItem && <Text>No queued generations</Text>}
      </Flex>
      <FormControl position="absolute" top={2} right={2} w="min-content">
        <FormLabel m={0}>Auto-switch</FormLabel>
        <Switch size="sm" isChecked={autoSwitch} onChange={onChangeAutoSwitch} />
      </FormControl>
      <Flex position="relative" w="full" maxW="full">
        <Flex ref={scrollableRef} gap={2} h={108} maxW="full" overflowX="scroll" flexShrink={0}>
          {items.map((item, i) => (
            <QueueItemStatusCard
              id={`queue-item-status-card-${item.item_id}`}
              key={item.item_id}
              item={item}
              number={i + 1}
              onSelectItem={onSelectItem}
              isSelected={selectedItemId === item.item_id}
              w={108}
              h={108}
              flexShrink={0}
            />
          ))}
        </Flex>
        <Box
          position="absolute"
          sx={scrollIndicatorSx}
          left={0}
          w={16}
          h="full"
          bg="linear-gradient(to right, var(--invoke-colors-base-900), transparent)"
          data-visible={canScrollLeft}
          transitionProperty="opacity"
          transitionDuration="0.3s"
          pointerEvents="none"
        />
        <Box
          position="absolute"
          sx={scrollIndicatorSx}
          right={0}
          w={16}
          h="full"
          bg="linear-gradient(to left, var(--invoke-colors-base-900), transparent)"
          data-visible={canScrollRight}
          transitionProperty="opacity"
          transitionDuration="0.3s"
          pointerEvents="none"
        />
      </Flex>
    </Flex>
  );
});
StagingArea.displayName = 'StagingArea';

const IMAGE_DTO_ERROR = Symbol('IMAGE_DTO_ERROR');

const useOutputImageDTO = (item: S['SessionQueueItem']) => {
  const [imageDTO, setImageDTO] = useState<ImageDTO | typeof IMAGE_DTO_ERROR | null>(null);
  const syncImageDTO = useCallback(async (item: S['SessionQueueItem']) => {
    const nodeId = Object.entries(item.session.source_prepared_mapping).find(([nodeId]) =>
      isCanvasOutputNodeId(nodeId)
    )?.[1][0];
    const output = nodeId ? item.session.results[nodeId] : undefined;

    if (!output) {
      return setImageDTO(null);
    }

    for (const [_name, value] of objectEntries(output)) {
      if (isImageField(value)) {
        const imageDTO = await getImageDTOSafe(value.image_name);
        if (imageDTO) {
          setImageDTO(imageDTO);
          $progressImages.setKey(item.session_id, undefined);
          return;
        }
      }
    }

    setImageDTO(IMAGE_DTO_ERROR);
  }, []);
  useEffect(() => {
    syncImageDTO(item);
  }, [item, syncImageDTO]);

  return imageDTO;
};

const QueueItemStatusCard = memo(
  ({
    item,
    isSelected,
    number,
    onSelectItem,
    ...rest
  }: {
    item: S['SessionQueueItem'];
    isSelected: boolean;
    number?: number;
    onSelectItem?: (item: S['SessionQueueItem']) => void;
  } & FlexProps) => {
    const onClick = useCallback(() => {
      onSelectItem?.(item);
    }, [item, onSelectItem]);
    return (
      <Flex
        role="button"
        pos="relative"
        borderWidth={1}
        borderRadius="base"
        alignItems="center"
        justifyContent="center"
        overflow="hidden"
        onClick={onClick}
        aspectRatio="1/1"
        borderColor={isSelected ? 'invokeBlue.300' : undefined}
        {...rest}
      >
        <QueueItemStatusCardContent item={item} />
        {number !== undefined && <Text position="absolute" top={0} left={1}>{`#${number}`}</Text>}
      </Flex>
    );
  }
);
QueueItemStatusCard.displayName = 'QueueItemStatusCard';

const QueueItemStatusCardContent = memo(({ item }: { item: S['SessionQueueItem'] }) => {
  const socket = useStore($socket);
  const [progressEvent, setProgressEvent] = useState<S['InvocationProgressEvent'] | null>(null);
  const [progressImage, setProgressImage] = useState<ProgressImage | null>(null);
  useEffect(() => {
    if (!socket) {
      return;
    }
    const onProgress = (data: S['InvocationProgressEvent']) => {
      if (data.session_id !== item.session_id) {
        return;
      }
      setProgressEvent(data);
      if (data.image) {
        setProgressImage(data.image);
      }
    };
    socket.on('invocation_progress', onProgress);

    return () => {
      socket.off('invocation_progress', onProgress);
    };
  }, [item.session_id, socket]);

  const imageDTO = useOutputImageDTO(item);

  if (item.status === 'pending') {
    return (
      <Text fontWeight="semibold" color="base.300">
        Pending
      </Text>
    );
  }
  if (item.status === 'canceled') {
    return (
      <Text fontWeight="semibold" color="warning.300">
        Canceled
      </Text>
    );
  }
  if (item.status === 'failed') {
    return (
      <Text fontWeight="semibold" color="error.300">
        Failed
      </Text>
    );
  }
  if (item.status === 'in_progress' || !imageDTO) {
    if (!progressImage) {
      return (
        <>
          <Text fontWeight="semibold" color="invokeBlue.300">
            In Progress
          </Text>
          <ProgressCircle data={progressEvent} />
        </>
      );
    }
    return (
      <>
        <Image objectFit="contain" maxH="full" maxW="full" src={progressImage.dataURL} width={progressImage.width} />
        <ProgressCircle data={progressEvent} />
      </>
    );
  }
  if (item.status === 'completed' && imageDTO && imageDTO !== IMAGE_DTO_ERROR) {
    return <Image objectFit="contain" maxH="full" maxW="full" src={imageDTO.image_url} width={imageDTO.width} />;
  }

  if (item.status === 'completed') {
    return (
      <Text fontWeight="semibold" color="error.300">
        Unable to get image
      </Text>
    );
  }
  assert<Equals<never, typeof item.status>>(false);
});
QueueItemStatusCardContent.displayName = 'QueueItemStatusCardContent';

const circleStyles: SystemStyleObject = {
  circle: {
    transitionProperty: 'none',
    transitionDuration: '0s',
  },
  position: 'absolute',
  top: 2,
  right: 2,
};

const ProgressCircle = ({ data }: { data?: S['InvocationProgressEvent'] | null }) => {
  return (
    <Tooltip label={data?.message ?? 'Generating'}>
      <CircularProgress
        size="14px"
        color="invokeBlue.500"
        thickness={14}
        isIndeterminate={!data || data.percentage === null}
        value={data?.percentage ? data.percentage * 100 : undefined}
        sx={circleStyles}
      />
    </Tooltip>
  );
};
ProgressCircle.displayName = 'ProgressCircle';

const QueueItemResultCard = memo(({ item }: { item: S['SessionQueueItem'] }) => {
  const imageName = useMemo(() => {
    const nodeId = Object.entries(item.session.source_prepared_mapping).find(([nodeId]) =>
      isCanvasOutputNodeId(nodeId)
    )?.[1][0];
    const output = nodeId ? item.session.results[nodeId] : undefined;
    if (!output) {
      return;
    }

    for (const [_name, value] of objectEntries(output)) {
      if (isImageField(value)) {
        return value.image_name;
      }
    }
  }, [item]);

  const { data: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);

  if (!imageDTO) {
    return <Text>Unknown output type</Text>;
  }

  return <Image objectFit="contain" maxH="full" maxW="full" src={imageDTO.image_url} width={imageDTO.width} />;
});
QueueItemResultCard.displayName = 'QueueItemResultCard';

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
                <Flex position="absolute" bottom={4} insetInlineEnd={4}>
                  <CanvasBusySpinner />
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
