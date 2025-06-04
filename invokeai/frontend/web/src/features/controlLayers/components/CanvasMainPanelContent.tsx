/* eslint-disable i18next/no-literal-string */
import type {
  ButtonGroupProps,
  CircularProgressProps,
  ImageProps,
  SystemStyleObject,
  TextProps,
} from '@invoke-ai/ui-library';
import {
  Button,
  ButtonGroup,
  CircularProgress,
  ContextMenu,
  Divider,
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
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
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
import { canvasSessionStarted, selectCanvasSession } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { newCanvasFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { isImageField } from 'features/nodes/types/common';
import { isCanvasOutputNodeId } from 'features/nodes/util/graph/graphBuilderUtils';
import { round } from 'lodash-es';
import { atom, type WritableAtom } from 'nanostores';
import type { ChangeEvent } from 'react';
import { createContext, memo, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { Trans, useTranslation } from 'react-i18next';
import { PiDotsThreeOutlineVerticalFill, PiUploadBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { useListAllQueueItemsQuery } from 'services/api/endpoints/queue';
import type { ImageDTO, S } from 'services/api/types';
import type { ProgressData } from 'services/events/stores';
import { $socket, setProgress, useProgressData } from 'services/events/stores';
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
  const session = useAppSelector(selectCanvasSession);

  if (session === null) {
    return <NoActiveSession />;
  }

  if (session.type === 'simple') {
    return <StagingAreaWrapper id={session.id} />;
  }

  if (session.type === 'advanced') {
    return <CanvasActiveSession />;
  }

  assert<Equals<never, typeof session>>(false, 'Unexpected session');
});
CanvasMainPanelContent.displayName = 'CanvasMainPanelContent';

const StagingAreaWrapper = memo(({ id }: { id: string }) => {
  const ctx = useMemo(
    () =>
      ({
        session: {
          type: 'simple',
          id,
        },
        $progressData: atom<Record<string, ProgressData>>({}),
      }) as const,
    [id]
  );

  return (
    <StagingContext.Provider value={ctx}>
      <StagingArea />
    </StagingContext.Provider>
  );
});
StagingAreaWrapper.displayName = 'StagingAreaWrapper';

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

const DROP_SHADOW = 'drop-shadow(0px 0px 4px rgb(0, 0, 0)) drop-shadow(0px 0px 4px rgba(0, 0, 0, 0.3))';

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

type StagingContextValue = {
  session:
    | {
        type: 'simple';
        id: string;
      }
    | {
        type: 'advanced';
        id: string;
      };
  $progressData: WritableAtom<Record<string, ProgressData>>;
};

const StagingContext = createContext<StagingContextValue | null>(null);

const useStagingContext = () => {
  const ctx = useContext(StagingContext);
  assert(ctx !== null, 'use in stg prov');
  return ctx;
};

const useStagingAreaKeyboardNav = (
  items: S['SessionQueueItem'][],
  selectedItemId: number | null,
  onSelectItemId: (item_id: number) => void
) => {
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
    onSelectItemId(nextItem.item_id);
  }, [items, onSelectItemId, selectedItemId]);
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
    onSelectItemId(prevItem.item_id);
  }, [items, onSelectItemId, selectedItemId]);

  const onFirst = useCallback(() => {
    const first = items.at(0);
    if (!first) {
      return;
    }
    onSelectItemId(first.item_id);
  }, [items, onSelectItemId]);
  const onLast = useCallback(() => {
    const last = items.at(-1);
    if (!last) {
      return;
    }
    onSelectItemId(last.item_id);
  }, [items, onSelectItemId]);

  useHotkeys('left', onPrev, { preventDefault: true });
  useHotkeys('right', onNext, { preventDefault: true });
  useHotkeys('meta+left', onFirst, { preventDefault: true });
  useHotkeys('meta+right', onLast, { preventDefault: true });
};

const LIST_ALL_OPTIONS = {
  selectFromResult: ({ data }) => {
    if (!data) {
      return { items: EMPTY_ARRAY };
    }
    return { items: data.filter(({ status }) => status !== 'canceled') };
  },
} satisfies Parameters<typeof useListAllQueueItemsQuery>[1];

const StagingArea = memo(() => {
  const ctx = useStagingContext();
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null);
  const [autoSwitch, setAutoSwitch] = useState(true);
  const { items } = useListAllQueueItemsQuery({ destination: ctx.session.id }, LIST_ALL_OPTIONS);
  const selectedItem = useMemo(() => {
    if (items.length === 0) {
      return null;
    }
    if (selectedItemId === null) {
      return null;
    }
    return items.find(({ item_id }) => item_id === selectedItemId) ?? null;
  }, [items, selectedItemId]);
  const selectedItemIndex = useMemo(() => {
    if (items.length === 0) {
      return null;
    }
    if (selectedItemId === null) {
      return null;
    }
    return items.findIndex(({ item_id }) => item_id === selectedItemId) ?? null;
  }, [items, selectedItemId]);

  const onSelectItemId = useCallback((item_id: number | null) => {
    setSelectedItemId(item_id);
    if (item_id !== null) {
      document.getElementById(getCardId(item_id))?.scrollIntoView();
    }
  }, []);

  useStagingAreaKeyboardNav(items, selectedItemId, onSelectItemId);

  useEffect(() => {
    if (items.length === 0) {
      onSelectItemId(null);
      return;
    }
    if (selectedItemId === null && items.length > 0) {
      onSelectItemId(items[0]?.item_id ?? null);
      return;
    }
  }, [items, onSelectItemId, selectedItem, selectedItemId]);

  const socket = useStore($socket);
  useEffect(() => {
    if (!socket) {
      return;
    }

    const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
      if (data.destination !== ctx.session.id) {
        return;
      }
      if (data.status === 'in_progress' && autoSwitch) {
        onSelectItemId(data.item_id);
      }
    };

    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [autoSwitch, ctx.$progressData, ctx.session.id, onSelectItemId, socket]);

  useEffect(() => {
    if (!socket) {
      return;
    }
    const onProgress = (data: S['InvocationProgressEvent']) => {
      if (data.destination !== ctx.session.id) {
        return;
      }
      setProgress(ctx.$progressData, data);
    };
    socket.on('invocation_progress', onProgress);

    return () => {
      socket.off('invocation_progress', onProgress);
    };
  }, [ctx.$progressData, ctx.session.id, socket]);

  return (
    <Flex flexDir="column" gap={2} w="full" h="full" minW={0} minH={0}>
      <StagingAreaHeader autoSwitch={autoSwitch} setAutoSwitch={setAutoSwitch} />
      <Divider />
      {items.length > 0 && (
        <StagingAreaContent
          items={items}
          selectedItem={selectedItem}
          selectedItemId={selectedItemId}
          selectedItemIndex={selectedItemIndex}
          onChangeAutoSwitch={setAutoSwitch}
          onSelectItemId={onSelectItemId}
        />
      )}
      {items.length === 0 && (
        <Flex w="full" h="full" alignItems="center" justifyContent="center">
          <Text>No generations</Text>
        </Flex>
      )}
    </Flex>
  );
});
StagingArea.displayName = 'StagingArea';

const StagingAreaContent = memo(
  ({
    items,
    selectedItem,
    selectedItemId,
    selectedItemIndex,
    onChangeAutoSwitch,
    onSelectItemId,
  }: {
    items: S['SessionQueueItem'][];
    selectedItem: S['SessionQueueItem'] | null;
    selectedItemId: number | null;
    selectedItemIndex: number | null;
    onChangeAutoSwitch: (autoSwitch: boolean) => void;
    onSelectItemId: (itemId: number) => void;
  }) => {
    return (
      <>
        <Flex position="relative" w="full" h="full" maxH="full" alignItems="center" justifyContent="center" minH={0}>
          {selectedItem && selectedItemIndex !== null && (
            <FullSizeQueueItem
              key={`${selectedItem.item_id}-full`}
              item={selectedItem}
              number={selectedItemIndex + 1}
            />
          )}
          {!selectedItem && <Text>No generation selected</Text>}
        </Flex>
        <Divider />
        <Flex position="relative" maxW="full" w="full" h={108}>
          <ScrollableContent overflowX="scroll" overflowY="hidden">
            <Flex gap={2} w="full" h="full">
              {items.map((item, i) => (
                <MiniQueueItem
                  key={`${item.item_id}-mini`}
                  item={item}
                  number={i + 1}
                  isSelected={selectedItemId === item.item_id}
                  onSelectItemId={onSelectItemId}
                  onChangeAutoSwitch={onChangeAutoSwitch}
                />
              ))}
            </Flex>
          </ScrollableContent>
        </Flex>
      </>
    );
  }
);
StagingAreaContent.displayName = 'StagingAreaContent';

const StagingAreaHeader = memo(
  ({ autoSwitch, setAutoSwitch }: { autoSwitch: boolean; setAutoSwitch: (autoSwitch: boolean) => void }) => {
    const dispatch = useAppDispatch();

    const startOver = useCallback(() => {
      dispatch(canvasSessionStarted({ sessionType: 'simple' }));
    }, [dispatch]);

    const onChangeAutoSwitch = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        setAutoSwitch(e.target.checked);
      },
      [setAutoSwitch]
    );

    return (
      <Flex gap={2} w="full" alignItems="center">
        <Text fontSize="lg" fontWeight="bold">
          Generations
        </Text>
        <Spacer />
        <FormControl w="min-content">
          <FormLabel m={0}>Auto-switch</FormLabel>
          <Switch size="sm" isChecked={autoSwitch} onChange={onChangeAutoSwitch} />
        </FormControl>
        <Button size="sm" variant="ghost" onClick={startOver}>
          Start Over
        </Button>
      </Flex>
    );
  }
);
StagingAreaHeader.displayName = 'StagingAreaHeader';

const miniQueueItemSx = {
  cursor: 'pointer',
  userSelect: 'none',
  pos: 'relative',
  alignItems: 'center',
  justifyContent: 'center',
  overflow: 'hidden',
  h: 'full',
  maxH: 'full',
  maxW: 'full',
  minW: 0,
  minH: 0,
  borderWidth: 1,
  borderRadius: 'base',
  '&[data-selected="true"]': {
    borderColor: 'invokeBlue.300',
  },
  aspectRatio: '1/1',
  flexShrink: 0,
} satisfies SystemStyleObject;

const getCardId = (item_id: number) => `queue-item-status-card-${item_id}`;

const getOutputImageName = (item: S['SessionQueueItem']) => {
  const nodeId = Object.entries(item.session.source_prepared_mapping).find(([nodeId]) =>
    isCanvasOutputNodeId(nodeId)
  )?.[1][0];
  const output = nodeId ? item.session.results[nodeId] : undefined;

  if (!output) {
    return null;
  }

  for (const [_name, value] of objectEntries(output)) {
    if (isImageField(value)) {
      return value.image_name;
    }
  }

  return null;
};

const useOutputImageDTO = (item: S['SessionQueueItem']) => {
  const outputImageName = useMemo(() => getOutputImageName(item), [item]);

  const { currentData: imageDTO } = useGetImageDTOQuery(outputImageName ?? skipToken);

  return imageDTO;
};

type MiniQueueItemProps = {
  item: S['SessionQueueItem'];
  number: number;
  isSelected: boolean;
  onSelectItemId: (item_id: number) => void;
  onChangeAutoSwitch: (autoSwitch: boolean) => void;
};

const MiniQueueItem = memo(({ item, isSelected, number, onSelectItemId, onChangeAutoSwitch }: MiniQueueItemProps) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const imageDTO = useOutputImageDTO(item);

  const onClick = useCallback(() => {
    onSelectItemId(item.item_id);
  }, [item.item_id, onSelectItemId]);

  const onDoubleClick = useCallback(() => {
    onChangeAutoSwitch(item.status === 'in_progress');
  }, [item.status, onChangeAutoSwitch]);

  const onLoad = useCallback(() => {
    setImageLoaded(true);
  }, []);

  return (
    <Flex
      id={getCardId(item.item_id)}
      sx={miniQueueItemSx}
      data-selected={isSelected}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      <ProgressLabel status={item.status} position="absolute" margin="auto" />
      {imageDTO && <DndImage imageDTO={imageDTO} asThumbnail onLoad={onLoad} />}
      {!imageLoaded && <ProgressImage session_id={item.session_id} position="absolute" />}
      <ItemNumber number={number} position="absolute" top={0} left={1} />
      <ProgressCircle session_id={item.session_id} status={item.status} position="absolute" top={1} right={2} />
    </Flex>
  );
});
MiniQueueItem.displayName = 'MiniQueueItem';

const fullSizeQueueItemSx = {
  cursor: 'pointer',
  userSelect: 'none',
  pos: 'relative',
  alignItems: 'center',
  justifyContent: 'center',
  overflow: 'hidden',
  h: 'full',
  w: 'full',
} satisfies SystemStyleObject;

type FullSizeQueueItemProps = {
  item: S['SessionQueueItem'];
  number: number;
};

const FullSizeQueueItem = memo(({ item, number }: FullSizeQueueItemProps) => {
  const imageDTO = useOutputImageDTO(item);
  const [imageLoaded, setImageLoaded] = useState(false);

  const onLoad = useCallback(() => {
    setImageLoaded(true);
  }, []);

  return (
    <Flex id={getCardId(item.item_id)} sx={fullSizeQueueItemSx}>
      <ProgressLabel status={item.status} position="absolute" margin="auto" />
      {imageDTO && <DndImage imageDTO={imageDTO} onLoad={onLoad} />}
      {!imageLoaded && <ProgressImage session_id={item.session_id} position="absolute" />}
      <ItemNumber number={number} position="absolute" top={1} left={2} />
      <ProgressMessage session_id={item.session_id} status={item.status} position="absolute" bottom={1} left={2} />
      <ProgressCircle session_id={item.session_id} status={item.status} position="absolute" top={1} right={2} />
    </Flex>
  );
});
FullSizeQueueItem.displayName = 'FullSizeQueueItem';

const ProgressImage = memo(({ session_id, ...rest }: { session_id: string } & ImageProps) => {
  const { $progressData } = useStagingContext();
  const { progressImage } = useProgressData($progressData, session_id);

  if (!progressImage) {
    return null;
  }

  return (
    <Image
      objectFit="contain"
      maxH="full"
      maxW="full"
      draggable={false}
      src={progressImage.dataURL}
      width={progressImage.width}
      height={progressImage.height}
      {...rest}
    />
  );
});
ProgressImage.displayName = 'ProgressImage';

const getMessage = (data: S['InvocationProgressEvent']) => {
  let message = data.message;
  if (data.percentage) {
    message += ` (${round(data.percentage * 100)}%)`;
  }
  return message;
};

const ItemNumber = memo(({ number, ...rest }: { number: number } & TextProps) => {
  return <Text pointerEvents="none" userSelect="none" filter={DROP_SHADOW} {...rest}>{`#${number}`}</Text>;
});
ItemNumber.displayName = 'ItemNumber';

const ProgressMessage = memo(
  ({ session_id, status, ...rest }: { session_id: string; status: S['SessionQueueItem']['status'] } & TextProps) => {
    const { $progressData } = useStagingContext();
    const { progressEvent } = useProgressData($progressData, session_id);

    if (status === 'completed' || status === 'failed' || status === 'canceled') {
      return null;
    }

    return (
      <Text pointerEvents="none" userSelect="none" filter={DROP_SHADOW} {...rest}>
        {progressEvent ? getMessage(progressEvent) : 'Waiting to start...'}
      </Text>
    );
  }
);
ProgressMessage.displayName = 'ProgressMessage';

const ProgressLabel = memo(({ status, ...rest }: { status: S['SessionQueueItem']['status'] } & TextProps) => {
  if (status === 'pending') {
    return (
      <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="base.300" {...rest}>
        Pending
      </Text>
    );
  }
  if (status === 'canceled') {
    return (
      <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="warning.300" {...rest}>
        Canceled
      </Text>
    );
  }
  if (status === 'failed') {
    return (
      <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="error.300" {...rest}>
        Failed
      </Text>
    );
  }

  if (status === 'in_progress') {
    return (
      <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="invokeBlue.300" {...rest}>
        In Progress
      </Text>
    );
  }

  return null;
});
ProgressLabel.displayName = 'ProgressLabel';

const circleStyles: SystemStyleObject = {
  circle: {
    transitionProperty: 'none',
    transitionDuration: '0s',
  },
  position: 'absolute',
  top: 2,
  right: 2,
};

const ProgressCircle = memo(
  ({
    session_id,
    status,
    ...rest
  }: { session_id: string; status: S['SessionQueueItem']['status'] } & CircularProgressProps) => {
    const { $progressData } = useStagingContext();
    const { progressEvent } = useProgressData($progressData, session_id);

    if (status !== 'in_progress') {
      return null;
    }

    return (
      <Tooltip label={progressEvent?.message ?? 'Generating'}>
        <CircularProgress
          size="14px"
          color="invokeBlue.500"
          thickness={14}
          isIndeterminate={!progressEvent || progressEvent.percentage === null}
          value={progressEvent?.percentage ? progressEvent.percentage * 100 : undefined}
          sx={circleStyles}
          {...rest}
        />
      </Tooltip>
    );
  }
);
ProgressCircle.displayName = 'ProgressCircle';

const ImageActions = memo(({ imageDTO, ...rest }: { imageDTO: ImageDTO } & ButtonGroupProps) => {
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
    <ButtonGroup isAttached={false} size="sm" {...rest}>
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
  );
});
ImageActions.displayName = 'ImageActions';

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
