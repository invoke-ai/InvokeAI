/* eslint-disable i18next/no-literal-string */

import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import { newCanvasFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { memo, useCallback } from 'react';
import { PiPencilBold } from 'react-icons/pi';
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

  return (
    <>
      <Icon as={PiPencilBold} boxSize={8} color="base.500" />
      <Heading size="sm">Edit Image</Heading>
      <Text color="base.300">Add an image to refine.</Text>
      <Flex w="full" justifyContent="flex-end">
        <UploadImageIconButton onUpload={onUpload} variant="link" h={8} />
      </Flex>
      <DndDropTarget dndTarget={newCanvasFromImageDndTarget} dndTargetData={dndTargetData} label="Drop" />
    </>
  );
});
InitialStateEditImageCard.displayName = 'InitialStateEditImageCard';
