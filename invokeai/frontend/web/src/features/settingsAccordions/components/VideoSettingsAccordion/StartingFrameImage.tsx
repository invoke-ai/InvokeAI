import { Box, Flex, FormLabel, Icon, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { objectEquals } from '@observ33r/object-equals';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import { ASPECT_RATIO_MAP } from 'features/controlLayers/store/types';
import { imageDTOToCroppableImage, imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { Editor } from 'features/cropper/lib/editor';
import { cropImageModalApi } from 'features/cropper/store';
import { videoFrameFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon, imageButtonSx } from 'features/dnd/DndImageIcon';
import {
  selectStartingFrameImage,
  selectVideoAspectRatio,
  selectVideoModelRequiresStartingFrame,
  startingFrameImageChanged,
} from 'features/parameters/store/videoSlice';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { PiArrowCounterClockwiseBold, PiCropBold, PiWarningBold } from 'react-icons/pi';
import { useImageDTO, useUploadImageMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

const dndTargetData = videoFrameFromImageDndTarget.getData({ frame: 'start' });

export const StartingFrameImage = () => {
  const dispatch = useAppDispatch();
  const requiresStartingFrame = useAppSelector(selectVideoModelRequiresStartingFrame);
  const startingFrameImage = useAppSelector(selectStartingFrameImage);
  const originalImageDTO = useImageDTO(startingFrameImage?.original.image_name);
  const croppedImageDTO = useImageDTO(startingFrameImage?.crop?.image.image_name);
  const videoAspectRatio = useAppSelector(selectVideoAspectRatio);
  const [uploadImage] = useUploadImageMutation({ fixedCacheKey: 'editorContainer' });

  const onReset = useCallback(() => {
    dispatch(startingFrameImageChanged(null));
  }, [dispatch]);

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      dispatch(startingFrameImageChanged(imageDTOToCroppableImage(imageDTO)));
    },
    [dispatch]
  );

  const edit = useCallback(() => {
    if (!originalImageDTO) {
      return;
    }

    // We will create a new editor instance each time the user wants to edit
    const editor = new Editor();

    // When the user applies the crop, we will upload the cropped image and store the applied crop box so if the user
    // re-opens the editor they see the same crop
    const onApplyCrop = async () => {
      const box = editor.getCropBox();
      if (objectEquals(box, startingFrameImage?.crop?.box)) {
        // If the box hasn't changed, don't do anything
        return;
      }
      if (!box || objectEquals(box, { x: 0, y: 0, width: originalImageDTO.width, height: originalImageDTO.height })) {
        // There is a crop applied but it is the whole iamge - revert to original image
        dispatch(startingFrameImageChanged(imageDTOToCroppableImage(originalImageDTO)));
        return;
      }
      const blob = await editor.exportImage('blob');
      const file = new File([blob], 'image.png', { type: 'image/png' });

      const newCroppedImageDTO = await uploadImage({
        file,
        is_intermediate: true,
        image_category: 'user',
      }).unwrap();

      dispatch(
        startingFrameImageChanged(
          imageDTOToCroppableImage(originalImageDTO, {
            image: imageDTOToImageWithDims(newCroppedImageDTO),
            box,
            ratio: editor.getCropAspectRatio(),
          })
        )
      );
    };

    const onReady = async () => {
      const initial = startingFrameImage?.crop
        ? { cropBox: startingFrameImage.crop.box, aspectRatio: startingFrameImage.crop.ratio }
        : undefined;
      // Load the image into the editor and open the modal once it's ready
      await editor.loadImage(originalImageDTO.image_url, initial);
    };

    cropImageModalApi.open({ editor, onApplyCrop, onReady });
  }, [dispatch, originalImageDTO, startingFrameImage?.crop, uploadImage]);

  const fitsCurrentAspectRatio = useMemo(() => {
    if (!originalImageDTO) {
      return true;
    }

    return originalImageDTO.width / originalImageDTO.height === ASPECT_RATIO_MAP[videoAspectRatio]?.ratio;
  }, [originalImageDTO, videoAspectRatio]);

  return (
    <Flex justifyContent="flex-start" flexDir="column" gap={2}>
      <FormLabel display="flex" alignItems="center" gap={2}>
        <Text>{t('parameters.startingFrameImage')}</Text>
        <Tooltip label={t('parameters.startingFrameImageAspectRatioWarning', { videoAspectRatio: videoAspectRatio })}>
          <Box>
            <Icon as={PiWarningBold} size={16} color="warning.500" />
          </Box>
        </Tooltip>
      </FormLabel>
      <Flex
        position="relative"
        w={36}
        h={36}
        alignItems="center"
        justifyContent="center"
        borderWidth={1}
        borderStyle="solid"
        borderColor={fitsCurrentAspectRatio ? 'base.500' : 'warning.500'}
      >
        {!originalImageDTO && (
          <UploadImageIconButton
            w="full"
            h="full"
            isError={requiresStartingFrame && !originalImageDTO}
            onUpload={onUpload}
            fontSize={36}
          />
        )}
        {originalImageDTO && (
          <>
            <DndImage
              imageDTO={croppedImageDTO ?? originalImageDTO}
              borderRadius="base"
              borderWidth={1}
              borderStyle="solid"
            />
            <Flex position="absolute" flexDir="column" top={1} insetInlineEnd={1} gap={1}>
              <DndImageIcon
                onClick={onReset}
                icon={<PiArrowCounterClockwiseBold size={16} />}
                tooltip={t('common.reset')}
              />
            </Flex>

            <Flex position="absolute" flexDir="column" top={1} insetInlineStart={1} gap={1}>
              <IconButton
                variant="link"
                sx={imageButtonSx}
                aria-label={t('common.crop')}
                onClick={edit}
                icon={<PiCropBold size={16} />}
                tooltip={t('common.crop')}
              />
            </Flex>

            <Text
              position="absolute"
              background="base.900"
              color="base.50"
              fontSize="sm"
              fontWeight="semibold"
              bottom={0}
              left={0}
              opacity={0.7}
              px={2}
              lineHeight={1.25}
              borderTopEndRadius="base"
              borderBottomStartRadius="base"
              pointerEvents="none"
            >{`${croppedImageDTO?.width ?? originalImageDTO.width}x${croppedImageDTO?.height ?? originalImageDTO.height}`}</Text>
          </>
        )}
        <DndDropTarget label="Drop" dndTarget={videoFrameFromImageDndTarget} dndTargetData={dndTargetData} />
      </Flex>
    </Flex>
  );
};
