import { Box, Flex, Spinner, SystemStyleObject } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import { memo, useCallback, useMemo, useState } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { controlNetImageChanged } from '../store/controlNetSlice';
import { PostUploadAction } from 'services/api/types';

type Props = {
  controlNetId: string;
  height: SystemStyleObject['h'];
};

const ControlNetImagePreview = (props: Props) => {
  const { height, controlNetId } = props;
  const dispatch = useAppDispatch();

  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlNet }) => {
          const { pendingControlImages } = controlNet;
          const {
            controlImage,
            processedControlImage,
            processorType,
            isEnabled,
          } = controlNet.controlNets[controlNetId];

          return {
            controlImageName: controlImage,
            processedControlImageName: processedControlImage,
            processorType,
            isEnabled,
            pendingControlImages,
          };
        },
        defaultSelectorOptions
      ),
    [controlNetId]
  );

  const {
    controlImageName,
    processedControlImageName,
    processorType,
    pendingControlImages,
    isEnabled,
  } = useAppSelector(selector);

  const [isMouseOverImage, setIsMouseOverImage] = useState(false);

  const {
    currentData: controlImage,
    isLoading: isLoadingControlImage,
    isError: isErrorControlImage,
    isSuccess: isSuccessControlImage,
  } = useGetImageDTOQuery(controlImageName ?? skipToken);

  const {
    currentData: processedControlImage,
    isLoading: isLoadingProcessedControlImage,
    isError: isErrorProcessedControlImage,
    isSuccess: isSuccessProcessedControlImage,
  } = useGetImageDTOQuery(processedControlImageName ?? skipToken);

  const handleResetControlImage = useCallback(() => {
    dispatch(controlNetImageChanged({ controlNetId, controlImage: null }));
  }, [controlNetId, dispatch]);
  const handleMouseEnter = useCallback(() => {
    setIsMouseOverImage(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsMouseOverImage(false);
  }, []);

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (controlImage) {
      return {
        id: controlNetId,
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO: controlImage },
      };
    }
  }, [controlImage, controlNetId]);

  const droppableData = useMemo<TypesafeDroppableData | undefined>(
    () => ({
      id: controlNetId,
      actionType: 'SET_CONTROLNET_IMAGE',
      context: { controlNetId },
    }),
    [controlNetId]
  );

  const postUploadAction = useMemo<PostUploadAction>(
    () => ({ type: 'SET_CONTROLNET_IMAGE', controlNetId }),
    [controlNetId]
  );

  const shouldShowProcessedImage =
    controlImage &&
    processedControlImage &&
    !isMouseOverImage &&
    !pendingControlImages.includes(controlNetId) &&
    processorType !== 'none';

  return (
    <Flex
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      sx={{
        position: 'relative',
        w: 'full',
        h: height,
        alignItems: 'center',
        justifyContent: 'center',
        pointerEvents: isEnabled ? 'auto' : 'none',
        opacity: isEnabled ? 1 : 0.5,
      }}
    >
      <IAIDndImage
        draggableData={draggableData}
        droppableData={droppableData}
        imageDTO={controlImage}
        isDropDisabled={shouldShowProcessedImage || !isEnabled}
        onClickReset={handleResetControlImage}
        postUploadAction={postUploadAction}
        resetTooltip="Reset Control Image"
        withResetIcon={Boolean(controlImage)}
      />
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineStart: 0,
          w: 'full',
          h: 'full',
          opacity: shouldShowProcessedImage ? 1 : 0,
          transitionProperty: 'common',
          transitionDuration: 'normal',
          pointerEvents: 'none',
        }}
      >
        <IAIDndImage
          draggableData={draggableData}
          droppableData={droppableData}
          imageDTO={processedControlImage}
          isUploadDisabled={true}
          isDropDisabled={!isEnabled}
          onClickReset={handleResetControlImage}
          resetTooltip="Reset Control Image"
          withResetIcon={Boolean(controlImage)}
        />
      </Box>
      {pendingControlImages.includes(controlNetId) && (
        <Flex
          sx={{
            position: 'absolute',
            top: 0,
            insetInlineStart: 0,
            w: 'full',
            h: 'full',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: 0.8,
            borderRadius: 'base',
            bg: 'base.400',
            _dark: {
              bg: 'base.900',
            },
          }}
        >
          <Spinner
            size="xl"
            sx={{ color: 'base.100', _dark: { color: 'base.400' } }}
          />
        </Flex>
      )}
    </Flex>
  );
};

export default memo(ControlNetImagePreview);
