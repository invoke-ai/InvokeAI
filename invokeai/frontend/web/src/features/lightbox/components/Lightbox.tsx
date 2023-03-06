import { Box, Flex, Grid } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import CurrentImageButtons from 'features/gallery/components/CurrentImageButtons';
import ImageGallery from 'features/gallery/components/ImageGallery';
import ImageMetadataViewer from 'features/gallery/components/ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from 'features/gallery/components/NextPrevImageButtons';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';
import { BiExit } from 'react-icons/bi';
import { TransformWrapper } from 'react-zoom-pan-pinch';
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
    <TransformWrapper
      centerOnInit
      minScale={0.1}
      initialPositionX={50}
      initialPositionY={50}
    >
      <Box
        sx={{
          width: '100%',
          height: '100%',
          overflow: 'hidden',
          position: 'absolute',
          insetInlineStart: 0,
          top: 0,
          zIndex: 30,
          animation: 'popIn 0.3s ease-in',
          bg: 'base.800',
        }}
      >
        <Flex
          sx={{
            flexDir: 'column',
            position: 'absolute',
            top: 4,
            insetInlineStart: 4,
            gap: 4,
            zIndex: 3,
          }}
        >
          <IAIIconButton
            icon={<BiExit />}
            aria-label="Exit Viewer"
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

        <Flex>
          <Grid
            sx={{
              overflow: 'hidden',
              gridTemplateColumns: 'auto max-content',
              placeItems: 'center',
              width: '100vw',
              height: '100vh',
              bg: 'base.850',
            }}
          >
            {viewerImageToDisplay && (
              <>
                <ReactPanZoomImage
                  rotation={rotation}
                  scaleX={scaleX}
                  scaleY={scaleY}
                  image={viewerImageToDisplay.url}
                  styleClass="lightbox-image"
                />
                {shouldShowImageDetails && (
                  <ImageMetadataViewer image={viewerImageToDisplay} />
                )}
              </>
            )}

            {!shouldShowImageDetails && (
              <Box
                sx={{
                  position: 'absolute',
                  top: 0,
                  insetInlineStart: 0,
                  w: `calc(100vw - ${8 * 2 * 4}px)`,
                  h: '100vh',
                  mx: 8,
                  pointerEvents: 'none',
                }}
              >
                <NextPrevImageButtons />
              </Box>
            )}

            <Box
              sx={{
                position: 'absolute',
                top: 4,
              }}
            >
              <CurrentImageButtons />
            </Box>
          </Grid>
          <ImageGallery />
        </Flex>
      </Box>
    </TransformWrapper>
  );
}
