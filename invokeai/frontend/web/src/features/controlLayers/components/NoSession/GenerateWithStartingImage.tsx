/* eslint-disable i18next/no-literal-string */

import { Button, Flex, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { newCanvasFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { memo, useMemo } from 'react';
import { Trans } from 'react-i18next';
import { PiUploadBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';
import type { Param0 } from 'tsafe';

const generateWithStartingImageDndTargetData = newCanvasFromImageDndTarget.getData({
  type: 'raster_layer',
  withResize: true,
});

export const GenerateWithStartingImage = memo(() => {
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
