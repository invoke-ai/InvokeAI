/* eslint-disable i18next/no-literal-string */

import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import { newCanvasFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { memo, useCallback } from 'react';
import { PiUserCircleGearBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

const NEW_CANVAS_OPTIONS = { type: 'reference_image' } as const;

const dndTargetData = newCanvasFromImageDndTarget.getData(NEW_CANVAS_OPTIONS);

export const InitialStateAddAStyleReference = memo(() => {
  const { getState, dispatch } = useAppStore();

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      newCanvasFromImage({ imageDTO, getState, dispatch, ...NEW_CANVAS_OPTIONS });
    },
    [dispatch, getState]
  );

  return (
    <>
      <Icon as={PiUserCircleGearBold} boxSize={8} color="base.500" />
      <Heading size="sm">Add a Style Reference</Heading>
      <Text color="base.300">Add an image to transfer its look.</Text>
      <Flex w="full" justifyContent="flex-end">
        <UploadImageIconButton onUpload={onUpload} variant="link" h={8} />
      </Flex>
      <DndDropTarget dndTarget={newCanvasFromImageDndTarget} dndTargetData={dndTargetData} label="Drop" />
    </>
  );
});
InitialStateAddAStyleReference.displayName = 'InitialStateAddAStyleReference';
