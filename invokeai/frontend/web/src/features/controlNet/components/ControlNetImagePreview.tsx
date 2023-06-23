import { memo, useCallback, useState } from 'react';
import { ImageDTO } from 'services/api/types';
import {
  ControlNetConfig,
  controlNetImageChanged,
  controlNetSelector,
} from '../store/controlNetSlice';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { Box, ChakraProps, Flex } from '@chakra-ui/react';
import IAIDndImage from 'common/components/IAIDndImage';
import { createSelector } from '@reduxjs/toolkit';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { AnimatePresence, motion } from 'framer-motion';
import { IAIImageLoadingFallback } from 'common/components/IAIImageFallback';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaUndo } from 'react-icons/fa';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { skipToken } from '@reduxjs/toolkit/dist/query';

const selector = createSelector(
  controlNetSelector,
  (controlNet) => {
    const { pendingControlImages } = controlNet;
    return { pendingControlImages };
  },
  defaultSelectorOptions
);

type Props = {
  controlNet: ControlNetConfig;
  imageSx?: ChakraProps['sx'];
};

const ControlNetImagePreview = (props: Props) => {
  const { imageSx } = props;
  const {
    controlNetId,
    controlImage: controlImageName,
    processedControlImage: processedControlImageName,
    processorType,
  } = props.controlNet;
  const dispatch = useAppDispatch();
  const { pendingControlImages } = useAppSelector(selector);

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

  const handleDrop = useCallback(
    (droppedImage: ImageDTO) => {
      if (controlImageName === droppedImage.image_name) {
        return;
      }
      setIsMouseOverImage(false);
      dispatch(
        controlNetImageChanged({
          controlNetId,
          controlImage: droppedImage.image_name,
        })
      );
    },
    [controlImageName, controlNetId, dispatch]
  );

  const handleResetControlImage = useCallback(() => {
    dispatch(controlNetImageChanged({ controlNetId, controlImage: null }));
  }, [controlNetId, dispatch]);
  const handleMouseEnter = useCallback(() => {
    setIsMouseOverImage(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsMouseOverImage(false);
  }, []);

  const shouldShowProcessedImageBackdrop =
    Number(controlImage?.width) > Number(processedControlImage?.width) ||
    Number(controlImage?.height) > Number(processedControlImage?.height);

  const shouldShowProcessedImage =
    controlImage &&
    processedControlImage &&
    !isMouseOverImage &&
    !pendingControlImages.includes(controlNetId) &&
    processorType !== 'none';

  return (
    <Box
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      sx={{ position: 'relative', w: 'full', h: 'full' }}
    >
      <IAIDndImage
        image={controlImage}
        onDrop={handleDrop}
        isDropDisabled={Boolean(
          processedControlImage && processorType !== 'none'
        )}
        isUploadDisabled={Boolean(controlImage)}
        postUploadAction={{ type: 'SET_CONTROLNET_IMAGE', controlNetId }}
        imageSx={imageSx}
      />
      <AnimatePresence>
        {shouldShowProcessedImage && (
          <motion.div
            style={{ width: '100%' }}
            initial={{
              opacity: 0,
            }}
            animate={{
              opacity: 1,
              transition: { duration: 0.1 },
            }}
            exit={{
              opacity: 0,
              transition: { duration: 0.1 },
            }}
          >
            <>
              {shouldShowProcessedImageBackdrop && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 0,
                    insetInlineStart: 0,
                    w: 'full',
                    h: 'full',
                    bg: 'base.900',
                    opacity: 0.7,
                  }}
                />
              )}
              <Box
                sx={{
                  position: 'absolute',
                  top: 0,
                  insetInlineStart: 0,
                  w: 'full',
                  h: 'full',
                }}
              >
                <IAIDndImage
                  image={processedControlImage}
                  onDrop={handleDrop}
                  payloadImage={controlImage}
                  isUploadDisabled={true}
                  imageSx={imageSx}
                />
              </Box>
            </>
          </motion.div>
        )}
      </AnimatePresence>
      {pendingControlImages.includes(controlNetId) && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            insetInlineStart: 0,
            w: 'full',
            h: 'full',
          }}
        >
          <IAIImageLoadingFallback />
        </Box>
      )}
      {controlImage && (
        <Flex sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}>
          <IAIIconButton
            aria-label="Reset Control Image"
            tooltip="Reset Control Image"
            size="sm"
            onClick={handleResetControlImage}
            icon={<FaUndo />}
            variant="link"
            sx={{
              p: 2,
              color: 'base.50',
            }}
          />
        </Flex>
      )}
    </Box>
  );
};

export default memo(ControlNetImagePreview);
