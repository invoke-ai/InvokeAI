import type { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { setCustomNativeDragPreview } from '@atlaskit/pragmatic-drag-and-drop/element/set-custom-native-drag-preview';
import { chakra, Flex } from '@invoke-ai/ui-library';
import type { SingleImageDndSourceData } from 'features/dnd/dnd';
import { DND_IMAGE_DRAG_PREVIEW_SIZE, preserveOffsetOnSourceFallbackCentered } from 'features/dnd/util';
import { memo } from 'react';
import { createPortal } from 'react-dom';
import type { ImageDTO } from 'services/api/types';
import type { Param0 } from 'tsafe';

const ChakraImg = chakra('img');

const DndDragPreviewSingleImage = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  return (
    <Flex w={DND_IMAGE_DRAG_PREVIEW_SIZE} h={DND_IMAGE_DRAG_PREVIEW_SIZE}>
      <ChakraImg
        margin="auto"
        maxW="full"
        maxH="full"
        objectFit="contain"
        borderRadius="base"
        src={imageDTO.thumbnail_url}
      />
    </Flex>
  );
});

DndDragPreviewSingleImage.displayName = 'DndDragPreviewSingleImage';

export type DndDragPreviewSingleImageState = {
  type: 'single-image';
  container: HTMLElement;
  imageDTO: ImageDTO;
};

export const createSingleImageDragPreview = (arg: DndDragPreviewSingleImageState) =>
  createPortal(<DndDragPreviewSingleImage imageDTO={arg.imageDTO} />, arg.container);

type SetSingleDragPreviewArg = {
  singleImageDndData: SingleImageDndSourceData;
  setDragPreviewState: (dragPreviewState: DndDragPreviewSingleImageState | null) => void;
  onGenerateDragPreviewArgs: Param0<Param0<typeof draggable>['onGenerateDragPreview']>;
};

export const setSingleImageDragPreview = ({
  singleImageDndData,
  onGenerateDragPreviewArgs,
  setDragPreviewState,
}: SetSingleDragPreviewArg) => {
  const { nativeSetDragImage, source, location } = onGenerateDragPreviewArgs;
  setCustomNativeDragPreview({
    render({ container }) {
      setDragPreviewState({ type: 'single-image', container, imageDTO: singleImageDndData.payload.imageDTO });
      return () => setDragPreviewState(null);
    },
    nativeSetDragImage,
    getOffset: preserveOffsetOnSourceFallbackCentered({
      element: source.element,
      input: location.current.input,
    }),
  });
};
