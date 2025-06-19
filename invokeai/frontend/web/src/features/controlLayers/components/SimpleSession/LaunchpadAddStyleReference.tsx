import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { LaunchpadButton } from 'features/controlLayers/components/SimpleSession/LaunchpadButton';
import { getDefaultRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { refImageAdded } from 'features/controlLayers/store/refImagesSlice';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { addGlobalReferenceImageDndTarget, newCanvasFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { memo, useMemo } from 'react';
import { PiUploadBold, PiUserCircleGearBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

const dndTargetData = addGlobalReferenceImageDndTarget.getData();

export const LaunchpadAddStyleReference = memo(() => {
  const { dispatch, getState } = useAppStore();

  const uploadOptions = useMemo(
    () =>
      ({
        onUpload: (imageDTO: ImageDTO) => {
          const config = getDefaultRefImageConfig(getState);
          config.image = imageDTOToImageWithDims(imageDTO);
          dispatch(refImageAdded({ overrides: { config } }));
        },
        allowMultiple: false,
      }) as const,
    [dispatch, getState]
  );

  const uploadApi = useImageUploadButton(uploadOptions);

  return (
    <LaunchpadButton {...uploadApi.getUploadButtonProps()} position="relative" gap={8}>
      <Icon as={PiUserCircleGearBold} boxSize={8} color="base.500" />
      <Flex flexDir="column" alignItems="flex-start" gap={2}>
        <Heading size="sm">Add a Style Reference</Heading>
        <Text color="base.300">Add an image to transfer its look.</Text>
      </Flex>
      <Flex position="absolute" right={3} bottom={3}>
        <PiUploadBold />
        <input {...uploadApi.getUploadInputProps()} />
      </Flex>
      <DndDropTarget dndTarget={newCanvasFromImageDndTarget} dndTargetData={dndTargetData} label="Drop" />
    </LaunchpadButton>
  );
});
LaunchpadAddStyleReference.displayName = 'LaunchpadAddStyleReference';
