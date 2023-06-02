import { memo, useCallback, useState } from 'react';
import { ImageDTO } from 'services/api';
import {
  controlNetImageChanged,
  controlNetSelector,
} from '../store/controlNetSlice';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { Box, Flex, Spinner } from '@chakra-ui/react';
import IAISelectableImage from './parameters/IAISelectableImage';
import { TbSquareToggle } from 'react-icons/tb';
import IAIIconButton from 'common/components/IAIIconButton';
import { createSelector } from '@reduxjs/toolkit';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { AnimatePresence, motion } from 'framer-motion';

const selector = createSelector(
  controlNetSelector,
  (controlNet) => {
    const { isProcessingControlImage } = controlNet;
    return { isProcessingControlImage };
  },
  defaultSelectorOptions
);

type Props = {
  controlNetId: string;
  controlImage: ImageDTO | null;
  processedControlImage: ImageDTO | null;
};

const ControlNetImagePreview = (props: Props) => {
  const { controlNetId, controlImage, processedControlImage } = props;
  const dispatch = useAppDispatch();
  const { isProcessingControlImage } = useAppSelector(selector);

  const [shouldShowProcessedImage, setShouldShowProcessedImage] =
    useState(true);

  const handleControlImageChanged = useCallback(
    (controlImage: ImageDTO) => {
      dispatch(controlNetImageChanged({ controlNetId, controlImage }));
    },
    [controlNetId, dispatch]
  );

  const handleControlImageReset = useCallback(() => {
    dispatch(controlNetImageChanged({ controlNetId, controlImage: null }));
  }, [controlNetId, dispatch]);

  const shouldShowProcessedImageBackdrop =
    Number(controlImage?.width) > Number(processedControlImage?.width) ||
    Number(controlImage?.height) > Number(processedControlImage?.height);

  return (
    <Box sx={{ position: 'relative', aspectRatio: '1/1' }}>
      <IAISelectableImage
        image={controlImage}
        onChange={handleControlImageChanged}
        onReset={handleControlImageReset}
        isDropDisabled={Boolean(processedControlImage)}
        fallback={<ProcessedImageFallback />}
        withResetIcon
        resetIconSize="sm"
      />
      <AnimatePresence>
        {controlImage &&
          processedControlImage &&
          shouldShowProcessedImage &&
          !isProcessingControlImage && (
            <motion.div
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
              <Box
                sx={{
                  position: 'absolute',
                  w: 'full',
                  h: 'full',
                  top: 0,
                  insetInlineStart: 0,
                }}
              >
                {shouldShowProcessedImageBackdrop && (
                  <Box
                    sx={{
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
                  <IAISelectableImage
                    image={processedControlImage}
                    onChange={handleControlImageChanged}
                    onReset={handleControlImageReset}
                    withResetIcon
                    resetIconSize="sm"
                    fallback={<ProcessedImageFallback />}
                  />
                </Box>
              </Box>
            </motion.div>
          )}
      </AnimatePresence>
      {isProcessingControlImage && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            insetInlineStart: 0,
            w: 'full',
            h: 'full',
          }}
        >
          <ProcessedImageFallback />
        </Box>
      )}
      {processedControlImage && !isProcessingControlImage && (
        <Box sx={{ position: 'absolute', bottom: 0, insetInlineEnd: 0, p: 2 }}>
          <IAIIconButton
            aria-label="Hide Preview"
            icon={<TbSquareToggle />}
            size="sm"
            onMouseOver={() => setShouldShowProcessedImage(false)}
            onMouseOut={() => setShouldShowProcessedImage(true)}
          />
        </Box>
      )}
    </Box>
  );
};

export default memo(ControlNetImagePreview);

const ProcessedImageFallback = () => (
  <Flex
    sx={{
      bg: 'base.900',
      opacity: 0.7,
      w: 'full',
      h: 'full',
      alignItems: 'center',
      justifyContent: 'center',
      borderRadius: 'base',
    }}
  >
    <Spinner size="xl" />
  </Flex>
);
