import {
  Box,
  ButtonGroup,
  Flex,
  FlexProps,
  Grid,
  Icon,
  Text,
  forwardRef,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import IAISlider from 'common/components/IAISlider';
import { imageGallerySelector } from 'features/gallery/store/gallerySelectors';
import {
  setCurrentCategory,
  setGalleryImageMinimumWidth,
  setGalleryImageObjectFit,
  setShouldAutoSwitchToNewImages,
  setShouldUseSingleGalleryColumn,
} from 'features/gallery/store/gallerySlice';
import { togglePinGalleryPanel } from 'features/ui/store/uiSlice';
import { useOverlayScrollbars } from 'overlayscrollbars-react';

import {
  ChangeEvent,
  PropsWithChildren,
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
import { FaImage, FaUser, FaWrench } from 'react-icons/fa';
import { MdPhotoLibrary } from 'react-icons/md';
import HoverableImage from './HoverableImage';

import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { resultsAdapter } from '../store/resultsSlice';
import {
  receivedResultImagesPage,
  receivedUploadImagesPage,
} from 'services/thunks/gallery';
import { uploadsAdapter } from '../store/uploadsSlice';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { Virtuoso, VirtuosoGrid } from 'react-virtuoso';

const GALLERY_SHOW_BUTTONS_MIN_WIDTH = 290;

const gallerySelector = createSelector(
  [
    (state: RootState) => state.uploads,
    (state: RootState) => state.results,
    (state: RootState) => state.gallery,
  ],
  (uploads, results, gallery) => {
    const { currentCategory } = gallery;

    return currentCategory === 'results'
      ? {
          images: resultsAdapter.getSelectors().selectAll(results),
          isLoading: results.isLoading,
          areMoreImagesAvailable: results.page < results.pages - 1,
        }
      : {
          images: uploadsAdapter.getSelectors().selectAll(uploads),
          isLoading: uploads.isLoading,
          areMoreImagesAvailable: uploads.page < uploads.pages - 1,
        };
  }
);

const ImageGalleryContent = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const resizeObserverRef = useRef<HTMLDivElement>(null);
  const [shouldShouldIconButtons, setShouldShouldIconButtons] = useState(true);
  const rootRef = useRef(null);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars({
    defer: true,
    options: {
      scrollbars: {
        visibility: 'auto',
        autoHide: 'leave',
        autoHideDelay: 1300,
        theme: 'os-theme-dark',
      },
      overflow: { x: 'hidden' },
    },
  });

  const {
    // images,
    currentCategory,
    shouldPinGallery,
    galleryImageMinimumWidth,
    galleryImageObjectFit,
    shouldAutoSwitchToNewImages,
    shouldUseSingleGalleryColumn,
    selectedImage,
  } = useAppSelector(imageGallerySelector);

  const { images, areMoreImagesAvailable, isLoading } =
    useAppSelector(gallerySelector);

  const handleClickLoadMore = () => {
    if (currentCategory === 'results') {
      dispatch(receivedResultImagesPage());
    }

    if (currentCategory === 'uploads') {
      dispatch(receivedUploadImagesPage());
    }
  };

  const handleChangeGalleryImageMinimumWidth = (v: number) => {
    dispatch(setGalleryImageMinimumWidth(v));
  };

  const handleSetShouldPinGallery = () => {
    dispatch(togglePinGalleryPanel());
    dispatch(requestCanvasRescale());
  };

  useEffect(() => {
    if (!resizeObserverRef.current) {
      return;
    }
    const resizeObserver = new ResizeObserver(() => {
      if (!resizeObserverRef.current) {
        return;
      }

      if (
        resizeObserverRef.current.clientWidth < GALLERY_SHOW_BUTTONS_MIN_WIDTH
      ) {
        setShouldShouldIconButtons(true);
        return;
      }

      setShouldShouldIconButtons(false);
    });
    resizeObserver.observe(resizeObserverRef.current);
    return () => resizeObserver.disconnect(); // clean up
  }, []);

  useEffect(() => {
    const { current: root } = rootRef;
    if (scroller && root) {
      initialize({
        target: root,
        elements: {
          viewport: scroller,
        },
      });
    }
    return () => osInstance()?.destroy();
  }, [scroller, initialize, osInstance]);

  const setScrollerRef = useCallback((ref: HTMLElement | Window | null) => {
    if (ref instanceof HTMLElement) {
      setScroller(ref);
    }
  }, []);

  return (
    <Flex flexDirection="column" w="full" h="full" gap={4}>
      <Flex
        ref={resizeObserverRef}
        alignItems="center"
        justifyContent="space-between"
      >
        <ButtonGroup
          size="sm"
          isAttached
          w="max-content"
          justifyContent="stretch"
        >
          {shouldShouldIconButtons ? (
            <>
              <IAIIconButton
                aria-label={t('gallery.showGenerations')}
                tooltip={t('gallery.showGenerations')}
                isChecked={currentCategory === 'results'}
                role="radio"
                icon={<FaImage />}
                onClick={() => dispatch(setCurrentCategory('results'))}
              />
              <IAIIconButton
                aria-label={t('gallery.showUploads')}
                tooltip={t('gallery.showUploads')}
                role="radio"
                isChecked={currentCategory === 'uploads'}
                icon={<FaUser />}
                onClick={() => dispatch(setCurrentCategory('uploads'))}
              />
            </>
          ) : (
            <>
              <IAIButton
                size="sm"
                isChecked={currentCategory === 'results'}
                onClick={() => dispatch(setCurrentCategory('results'))}
                flexGrow={1}
              >
                {t('gallery.generations')}
              </IAIButton>
              <IAIButton
                size="sm"
                isChecked={currentCategory === 'uploads'}
                onClick={() => dispatch(setCurrentCategory('uploads'))}
                flexGrow={1}
              >
                {t('gallery.uploads')}
              </IAIButton>
            </>
          )}
        </ButtonGroup>

        <Flex gap={2}>
          <IAIPopover
            triggerComponent={
              <IAIIconButton
                size="sm"
                aria-label={t('gallery.gallerySettings')}
                icon={<FaWrench />}
              />
            }
          >
            <Flex direction="column" gap={2}>
              <IAISlider
                value={galleryImageMinimumWidth}
                onChange={handleChangeGalleryImageMinimumWidth}
                min={32}
                max={256}
                hideTooltip={true}
                label={t('gallery.galleryImageSize')}
                withReset
                handleReset={() => dispatch(setGalleryImageMinimumWidth(64))}
              />
              <IAICheckbox
                label={t('gallery.maintainAspectRatio')}
                isChecked={galleryImageObjectFit === 'contain'}
                onChange={() =>
                  dispatch(
                    setGalleryImageObjectFit(
                      galleryImageObjectFit === 'contain' ? 'cover' : 'contain'
                    )
                  )
                }
              />
              <IAICheckbox
                label={t('gallery.autoSwitchNewImages')}
                isChecked={shouldAutoSwitchToNewImages}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  dispatch(setShouldAutoSwitchToNewImages(e.target.checked))
                }
              />
              <IAICheckbox
                label={t('gallery.singleColumnLayout')}
                isChecked={shouldUseSingleGalleryColumn}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  dispatch(setShouldUseSingleGalleryColumn(e.target.checked))
                }
              />
            </Flex>
          </IAIPopover>

          <IAIIconButton
            size="sm"
            aria-label={t('gallery.pinGallery')}
            tooltip={`${t('gallery.pinGallery')} (Shift+G)`}
            onClick={handleSetShouldPinGallery}
            icon={shouldPinGallery ? <BsPinAngleFill /> : <BsPinAngle />}
          />
        </Flex>
      </Flex>
      <Flex direction="column" gap={2} h="full">
        {images.length || areMoreImagesAvailable ? (
          <>
            <Box ref={rootRef} data-overlayscrollbars="" h="100%">
              {shouldUseSingleGalleryColumn ? (
                <Virtuoso
                  style={{ height: '100%' }}
                  data={images}
                  scrollerRef={(ref) => setScrollerRef(ref)}
                  itemContent={(index, image) => {
                    const { name } = image;
                    const isSelected = selectedImage?.name === name;

                    return (
                      <Flex sx={{ pb: 2 }}>
                        <HoverableImage
                          key={`${name}-${image.thumbnail}`}
                          image={image}
                          isSelected={isSelected}
                        />
                      </Flex>
                    );
                  }}
                />
              ) : (
                <VirtuosoGrid
                  style={{ height: '100%' }}
                  data={images}
                  components={{
                    Item: ItemContainer,
                    List: ListContainer,
                  }}
                  scrollerRef={setScroller}
                  itemContent={(index, image) => {
                    const { name } = image;
                    const isSelected = selectedImage?.name === name;

                    return (
                      <HoverableImage
                        key={`${name}-${image.thumbnail}`}
                        image={image}
                        isSelected={isSelected}
                      />
                    );
                  }}
                />
              )}
            </Box>
            <IAIButton
              onClick={handleClickLoadMore}
              isDisabled={!areMoreImagesAvailable}
              isLoading={isLoading}
              flexShrink={0}
            >
              {areMoreImagesAvailable
                ? t('gallery.loadMore')
                : t('gallery.allImagesLoaded')}
            </IAIButton>
          </>
        ) : (
          <Flex
            sx={{
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 2,
              padding: 8,
              h: '100%',
              w: '100%',
              color: 'base.500',
            }}
          >
            <Icon
              as={MdPhotoLibrary}
              sx={{
                w: 16,
                h: 16,
              }}
            />
            <Text textAlign="center">{t('gallery.noImagesInGallery')}</Text>
          </Flex>
        )}
      </Flex>
    </Flex>
  );
};

type ItemContainerProps = PropsWithChildren & FlexProps;
const ItemContainer = forwardRef((props: ItemContainerProps, ref) => (
  <Box className="item-container" ref={ref}>
    {props.children}
  </Box>
));

type ListContainerProps = PropsWithChildren & FlexProps;
const ListContainer = forwardRef((props: ListContainerProps, ref) => {
  const galleryImageMinimumWidth = useAppSelector(
    (state: RootState) => state.gallery.galleryImageMinimumWidth
  );

  return (
    <Grid
      {...props}
      className="list-container"
      ref={ref}
      sx={{
        gap: 2,
        gridTemplateColumns: `repeat(auto-fit, minmax(${galleryImageMinimumWidth}px, 1fr));`,
      }}
    >
      {props.children}
    </Grid>
  );
});

export default memo(ImageGalleryContent);
