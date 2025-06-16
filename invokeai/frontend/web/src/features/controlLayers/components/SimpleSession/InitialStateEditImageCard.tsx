import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { InitialStateButtonGridItem } from 'features/controlLayers/components/SimpleSession/InitialStateButtonGridItem';
import { newCanvasFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { memo, useCallback } from 'react';
import { PiPencilBold, PiUploadBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

const NEW_CANVAS_OPTIONS = { type: 'raster_layer', withInpaintMask: true } as const;

const dndTargetData = newCanvasFromImageDndTarget.getData(NEW_CANVAS_OPTIONS);

export const InitialStateEditImageCard = memo(() => {
  const { getState, dispatch } = useAppStore();

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      newCanvasFromImage({ imageDTO, getState, dispatch, ...NEW_CANVAS_OPTIONS });
    },
    [dispatch, getState]
  );
  const uploadApi = useImageUploadButton({ allowMultiple: false, onUpload });

  return (
    <InitialStateButtonGridItem {...uploadApi.getUploadButtonProps()}>
      <Icon as={PiPencilBold} boxSize={8} color="base.500" />
      <Heading size="sm">Edit Image</Heading>
      <Text color="base.300">Add an image to refine.</Text>
      <Flex w="full" justifyContent="flex-end" p={2}>
        <PiUploadBold />
        <input {...uploadApi.getUploadInputProps()} />
      </Flex>
      <DndDropTarget dndTarget={newCanvasFromImageDndTarget} dndTargetData={dndTargetData} label="Drop" />
    </InitialStateButtonGridItem>
  );
});
InitialStateEditImageCard.displayName = 'InitialStateEditImageCard';
