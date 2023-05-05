import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import CurrentImageButtons from 'features/gallery/components/CurrentImageButtons';
import ImageMetadataViewer from 'features/gallery/components/ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from 'features/gallery/components/NextPrevImageButtons';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { AnimatePresence, motion } from 'framer-motion';
import { isEqual } from 'lodash-es';
import { useHotkeys } from 'react-hotkeys-hook';
import { BiExit } from 'react-icons/bi';
import { TransformWrapper } from 'react-zoom-pan-pinch';
import { PROGRESS_BAR_THICKNESS } from 'theme/util/constants';
import useImageTransform from '../hooks/useImageTransform';
import ReactPanZoomButtons from './ReactPanZoomButtons';
import ReactPanZoomImage from './ReactPanZoomImage';

export const lightboxSelector = createSelector(
  [gallerySelector, uiSelector],
  (gallery, ui) => {
    const { currentImage } = gallery;
    const { shouldShowImageDetails } = ui;

    return {
      viewerImageToDisplay: currentImage,
      shouldShowImageDetails,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export default function Lightbox() {
  const dispatch = useAppDispatch();
  const isLightBoxOpen = useAppSelector(
    (state: RootState) => state.lightbox.isLightboxOpen
  );

  const {
    rotation,
    scaleX,
    scaleY,
    flipHorizontally,
    flipVertically,
    rotateCounterClockwise,
    rotateClockwise,
    reset,
  } = useImageTransform();

  const { viewerImageToDisplay, shouldShowImageDetails } =
    useAppSelector(lightboxSelector);

  useHotkeys(
    'Esc',
    () => {
      if (isLightBoxOpen) dispatch(setIsLightboxOpen(false));
    },
    [isLightBoxOpen]
  );

  return (
    <AnimatePresence>
      {isLightBoxOpen && (
        <motion.div
          key="lightbox"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15, ease: 'easeInOut' }}
          style={{
            display: 'flex',
            width: '100vw',
            height: `calc(100vh - ${PROGRESS_BAR_THICKNESS * 4}px)`,
            position: 'fixed',
            top: `${PROGRESS_BAR_THICKNESS * 4}px`,
            background: 'var(--invokeai-colors-base-900)',
            zIndex: 99,
          }}
        >
          <TransformWrapper
            centerOnInit
            minScale={0.1}
            initialPositionX={50}
            initialPositionY={50}
          >
            <Flex
              sx={{
                flexDir: 'column',
                position: 'absolute',
                insetInlineStart: 4,
                gap: 4,
                zIndex: 3,
                top: 4,
              }}
            >
              <IAIIconButton
                icon={<BiExit />}
                aria-label="Exit Viewer"
                className="lightbox-close-btn"
                onClick={() => {
                  dispatch(setIsLightboxOpen(false));
                }}
                fontSize={20}
              />
              <ReactPanZoomButtons
                flipHorizontally={flipHorizontally}
                flipVertically={flipVertically}
                rotateCounterClockwise={rotateCounterClockwise}
                rotateClockwise={rotateClockwise}
                reset={reset}
              />
            </Flex>
            <Flex
              sx={{
                position: 'absolute',
                top: 4,
                zIndex: 3,
                insetInlineStart: '50%',
                transform: 'translate(-50%, 0)',
              }}
            >
              <CurrentImageButtons />
            </Flex>

            {viewerImageToDisplay && (
              <>
                <ReactPanZoomImage
                  rotation={rotation}
                  scaleX={scaleX}
                  scaleY={scaleY}
                  image={viewerImageToDisplay}
                  styleClass="lightbox-image"
                />
                {shouldShowImageDetails && (
                  <ImageMetadataViewer image={viewerImageToDisplay} />
                )}

                {!shouldShowImageDetails && (
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 0,
                      insetInlineStart: 0,
                      w: '100vw',
                      h: '100vh',
                      px: 16,
                      pointerEvents: 'none',
                    }}
                  >
                    <NextPrevImageButtons />
                  </Box>
                )}
              </>
            )}
          </TransformWrapper>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
