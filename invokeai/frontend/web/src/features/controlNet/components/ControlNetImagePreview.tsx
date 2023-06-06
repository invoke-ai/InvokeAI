import { memo, useCallback, useRef } from 'react';
import { ImageDTO } from 'services/api';
import {
  ControlNetConfig,
  controlNetImageChanged,
  controlNetSelector,
} from '../store/controlNetSlice';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { Box } from '@chakra-ui/react';
import IAIDndImage from 'common/components/IAIDndImage';
import { createSelector } from '@reduxjs/toolkit';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { AnimatePresence, motion } from 'framer-motion';
import { IAIImageFallback } from 'common/components/IAIImageFallback';
import { useHoverDirty } from 'react-use';

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
};

const ControlNetImagePreview = (props: Props) => {
  const { controlNetId, controlImage, processedControlImage, processorType } =
    props.controlNet;
  const dispatch = useAppDispatch();
  const { pendingControlImages } = useAppSelector(selector);
  const containerRef = useRef<HTMLDivElement>(null);

  const isMouseOverImage = useHoverDirty(containerRef);

  const handleDrop = useCallback(
    (droppedImage: ImageDTO) => {
      if (controlImage?.image_name === droppedImage.image_name) {
        return;
      }
      dispatch(
        controlNetImageChanged({ controlNetId, controlImage: droppedImage })
      );
    },
    [controlImage, controlNetId, dispatch]
  );

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
      ref={containerRef}
      sx={{ position: 'relative', w: 'full', h: 'full', aspectRatio: '1/1' }}
    >
      <IAIDndImage
        image={controlImage}
        onDrop={handleDrop}
        isDropDisabled={Boolean(
          processedControlImage && processorType !== 'none'
        )}
      />
      <AnimatePresence>
        {shouldShowProcessedImage && (
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
                <IAIDndImage
                  image={processedControlImage}
                  onDrop={handleDrop}
                  payloadImage={controlImage}
                />
              </Box>
            </Box>
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
          <IAIImageFallback />
        </Box>
      )}
    </Box>
  );
};

export default memo(ControlNetImagePreview);
