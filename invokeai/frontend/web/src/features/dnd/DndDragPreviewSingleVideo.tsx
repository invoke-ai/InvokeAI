import type { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { setCustomNativeDragPreview } from '@atlaskit/pragmatic-drag-and-drop/element/set-custom-native-drag-preview';
import { chakra, Flex } from '@invoke-ai/ui-library';
import type { SingleVideoDndSourceData } from 'features/dnd/dnd';
import { DND_IMAGE_DRAG_PREVIEW_SIZE, preserveOffsetOnSourceFallbackCentered } from 'features/dnd/util';
import { GalleryVideoPlaceholder } from 'features/gallery/components/ImageGrid/GalleryVideoPlaceholder';
import { memo } from 'react';
import { createPortal } from 'react-dom';
import type { VideoDTO } from 'services/api/types';
import type { Param0 } from 'tsafe';

const ChakraImg = chakra('img');

const DndDragPreviewSingleVideo = memo(({ videoDTO }: { videoDTO: VideoDTO }) => {
  return (
    <Flex position="relative" w={DND_IMAGE_DRAG_PREVIEW_SIZE} h={DND_IMAGE_DRAG_PREVIEW_SIZE}>
      <GalleryVideoPlaceholder />
      <ChakraImg
        position="absolute"
        margin="auto"
        maxW="full"
        maxH="full"
        objectFit="contain"
        borderRadius="base"
        borderWidth={2}
        borderColor="invokeBlue.300"
        borderStyle="solid"
        cursor="grabbing"
        src={videoDTO.thumbnail_url}
      />
    </Flex>
  );
});

DndDragPreviewSingleVideo.displayName = 'DndDragPreviewSingleVideo';

export type DndDragPreviewSingleVideoState = {
  type: 'single-video';
  container: HTMLElement;
  videoDTO: VideoDTO;
};

export const createSingleVideoDragPreview = (arg: DndDragPreviewSingleVideoState) =>
  createPortal(<DndDragPreviewSingleVideo videoDTO={arg.videoDTO} />, arg.container);

type SetSingleDragPreviewArg = {
  singleVideoDndData: SingleVideoDndSourceData;
  setDragPreviewState: (dragPreviewState: DndDragPreviewSingleVideoState | null) => void;
  onGenerateDragPreviewArgs: Param0<Param0<typeof draggable>['onGenerateDragPreview']>;
};

export const setSingleVideoDragPreview = ({
  singleVideoDndData,
  onGenerateDragPreviewArgs,
  setDragPreviewState,
}: SetSingleDragPreviewArg) => {
  const { nativeSetDragImage, source, location } = onGenerateDragPreviewArgs;
  setCustomNativeDragPreview({
    render({ container }) {
      setDragPreviewState({ type: 'single-video', container, videoDTO: singleVideoDndData.payload.videoDTO });
      return () => setDragPreviewState(null);
    },
    nativeSetDragImage,
    getOffset: preserveOffsetOnSourceFallbackCentered({
      element: source.element,
      input: location.current.input,
    }),
  });
};
