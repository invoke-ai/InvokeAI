import { Box, Flex, Spinner, useShiftModifier } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { bboxHeightChanged, bboxWidthChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectOptimalDimension } from 'features/controlLayers/store/selectors';
import type { CanvasControlAdapterState } from 'features/controlLayers/store/types';
import type { ImageDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { calculateNewSize } from 'features/parameters/components/DocumentSize/calculateNewSize';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiFloppyDiskBold, PiRulerBold } from 'react-icons/pi';
import {
  useAddImageToBoardMutation,
  useChangeImageIsIntermediateMutation,
  useGetImageDTOQuery,
  useRemoveImageFromBoardMutation,
} from 'services/api/endpoints/images';
import type { ImageDTO, PostUploadAction } from 'services/api/types';

type Props = {
  controlAdapter: CanvasControlAdapterState;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  droppableData: TypesafeDroppableData;
  postUploadAction: PostUploadAction;
  onErrorLoadingImage: () => void;
  onErrorLoadingProcessedImage: () => void;
};

export const CAImagePreview = memo(
  ({
    controlAdapter,
    onChangeImage,
    droppableData,
    postUploadAction,
    onErrorLoadingImage,
    onErrorLoadingProcessedImage,
  }: Props) => {
    const { t } = useTranslation();
    const dispatch = useAppDispatch();
    const autoAddBoardId = useAppSelector((s) => s.gallery.autoAddBoardId);
    const isConnected = useAppSelector((s) => s.system.isConnected);
    const optimalDimension = useAppSelector(selectOptimalDimension);
    const shift = useShiftModifier();

    const [isMouseOverImage, setIsMouseOverImage] = useState(false);

    const { currentData: controlImage, isError: isErrorControlImage } = useGetImageDTOQuery(
      controlAdapter.imageObject?.image.name ?? skipToken
    );
    const { currentData: processedControlImage, isError: isErrorProcessedControlImage } = useGetImageDTOQuery(
      controlAdapter.processedImageObject?.image.name ?? skipToken
    );

    const [changeIsIntermediate] = useChangeImageIsIntermediateMutation();
    const [addToBoard] = useAddImageToBoardMutation();
    const [removeFromBoard] = useRemoveImageFromBoardMutation();
    const handleResetControlImage = useCallback(() => {
      onChangeImage(null);
    }, [onChangeImage]);

    const handleSaveControlImage = useCallback(async () => {
      if (!processedControlImage) {
        return;
      }

      await changeIsIntermediate({
        imageDTO: processedControlImage,
        is_intermediate: false,
      }).unwrap();

      if (autoAddBoardId !== 'none') {
        addToBoard({
          imageDTO: processedControlImage,
          board_id: autoAddBoardId,
        });
      } else {
        removeFromBoard({ imageDTO: processedControlImage });
      }
    }, [processedControlImage, changeIsIntermediate, autoAddBoardId, addToBoard, removeFromBoard]);

    const handleSetControlImageToDimensions = useCallback(() => {
      if (!controlImage) {
        return;
      }

      const options = { updateAspectRatio: true, clamp: true };

      if (shift) {
        const { width, height } = controlImage;
        dispatch(bboxWidthChanged({ width, ...options }));
        dispatch(bboxHeightChanged({ height, ...options }));
      } else {
        const { width, height } = calculateNewSize(
          controlImage.width / controlImage.height,
          optimalDimension * optimalDimension
        );
        dispatch(bboxWidthChanged({ width, ...options }));
        dispatch(bboxHeightChanged({ height, ...options }));
      }
    }, [controlImage, dispatch, optimalDimension, shift]);

    const handleMouseEnter = useCallback(() => {
      setIsMouseOverImage(true);
    }, []);

    const handleMouseLeave = useCallback(() => {
      setIsMouseOverImage(false);
    }, []);

    const draggableData = useMemo<ImageDraggableData | undefined>(() => {
      if (controlImage) {
        return {
          id: controlAdapter.id,
          payloadType: 'IMAGE_DTO',
          payload: { imageDTO: controlImage },
        };
      }
    }, [controlImage, controlAdapter.id]);

    const shouldShowProcessedImage =
      controlImage &&
      processedControlImage &&
      !isMouseOverImage &&
      !controlAdapter.processorPendingBatchId &&
      controlAdapter.processorConfig !== null;

    useEffect(() => {
      if (!isConnected) {
        return;
      }
      if (isErrorControlImage) {
        onErrorLoadingImage();
      }
      if (isErrorProcessedControlImage) {
        onErrorLoadingProcessedImage();
      }
    }, [
      handleResetControlImage,
      isConnected,
      isErrorControlImage,
      isErrorProcessedControlImage,
      onErrorLoadingImage,
      onErrorLoadingProcessedImage,
    ]);

    return (
      <Flex
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        position="relative"
        w={36}
        h={36}
        alignItems="center"
        justifyContent="center"
      >
        <IAIDndImage
          draggableData={draggableData}
          droppableData={droppableData}
          imageDTO={controlImage}
          isDropDisabled={shouldShowProcessedImage}
          postUploadAction={postUploadAction}
        />

        <Box
          position="absolute"
          top={0}
          insetInlineStart={0}
          w="full"
          h="full"
          opacity={shouldShowProcessedImage ? 1 : 0}
          transitionProperty="common"
          transitionDuration="normal"
          pointerEvents="none"
        >
          <IAIDndImage
            draggableData={draggableData}
            droppableData={droppableData}
            imageDTO={processedControlImage}
            isUploadDisabled={true}
            onError={handleResetControlImage}
          />
        </Box>

        {controlImage && (
          <Flex position="absolute" flexDir="column" top={1} insetInlineEnd={1} gap={1}>
            <IAIDndImageIcon
              onClick={handleResetControlImage}
              icon={<PiArrowCounterClockwiseBold size={16} />}
              tooltip={t('controlnet.resetControlImage')}
            />
            <IAIDndImageIcon
              onClick={handleSaveControlImage}
              icon={<PiFloppyDiskBold size={16} />}
              tooltip={t('controlnet.saveControlImage')}
            />
            <IAIDndImageIcon
              onClick={handleSetControlImageToDimensions}
              icon={<PiRulerBold size={16} />}
              tooltip={
                shift ? t('controlnet.setControlImageDimensionsForce') : t('controlnet.setControlImageDimensions')
              }
            />
          </Flex>
        )}

        {controlAdapter.processorPendingBatchId !== null && (
          <Flex
            position="absolute"
            top={0}
            insetInlineStart={0}
            w="full"
            h="full"
            alignItems="center"
            justifyContent="center"
            opacity={0.8}
            borderRadius="base"
            bg="base.900"
          >
            <Spinner size="xl" color="base.400" />
          </Flex>
        )}
      </Flex>
    );
  }
);

CAImagePreview.displayName = 'CAImagePreview';
