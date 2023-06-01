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
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import {
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
  useMemo,
  useRef,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
import { FaImage, FaServer, FaWrench } from 'react-icons/fa';
import { MdPhotoLibrary } from 'react-icons/md';
import HoverableImage from './HoverableImage';

import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { Virtuoso, VirtuosoGrid } from 'react-virtuoso';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { uiSelector } from 'features/ui/store/uiSelectors';
import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  imageCategoriesChanged,
  selectImagesAll,
} from '../store/imagesSlice';
import { receivedPageOfImages } from 'services/thunks/image';

const categorySelector = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { images } = state;
    const { categories } = images;

    const allImages = selectImagesAll(state);
    const filteredImages = allImages.filter((i) =>
      categories.includes(i.image_category)
    );

    return {
      images: filteredImages,
      isLoading: images.isLoading,
      areMoreImagesAvailable: filteredImages.length < images.total,
      categories: images.categories,
    };
  },
  defaultSelectorOptions
);

const mainSelector = createSelector(
  [gallerySelector, uiSelector],
  (gallery, ui) => {
    const {
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      shouldAutoSwitchToNewImages,
      shouldUseSingleGalleryColumn,
      selectedImage,
    } = gallery;

    const { shouldPinGallery } = ui;

    return {
      shouldPinGallery,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      shouldAutoSwitchToNewImages,
      shouldUseSingleGalleryColumn,
      selectedImage,
    };
  },
  defaultSelectorOptions
);

const ImageGalleryContent = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const resizeObserverRef = useRef<HTMLDivElement>(null);
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
    shouldPinGallery,
    galleryImageMinimumWidth,
    galleryImageObjectFit,
    shouldAutoSwitchToNewImages,
    shouldUseSingleGalleryColumn,
    selectedImage,
  } = useAppSelector(mainSelector);

  const { images, areMoreImagesAvailable, isLoading, categories } =
    useAppSelector(categorySelector);

  const handleLoadMoreImages = useCallback(() => {
    dispatch(receivedPageOfImages());
  }, [dispatch]);

  const handleEndReached = useMemo(() => {
    if (areMoreImagesAvailable && !isLoading) {
      return handleLoadMoreImages;
    }
    return undefined;
  }, [areMoreImagesAvailable, handleLoadMoreImages, isLoading]);

  const handleChangeGalleryImageMinimumWidth = (v: number) => {
    dispatch(setGalleryImageMinimumWidth(v));
  };

  const handleSetShouldPinGallery = () => {
    dispatch(togglePinGalleryPanel());
    dispatch(requestCanvasRescale());
  };

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

  const handleClickImagesCategory = useCallback(() => {
    dispatch(imageCategoriesChanged(IMAGE_CATEGORIES));
  }, [dispatch]);

  const handleClickAssetsCategory = useCallback(() => {
    dispatch(imageCategoriesChanged(ASSETS_CATEGORIES));
  }, [dispatch]);

  return (
    <Flex
      sx={{
        gap: 2,
        flexDirection: 'column',
        h: 'full',
        w: 'full',
        borderRadius: 'base',
      }}
    >
      <Flex
        ref={resizeObserverRef}
        alignItems="center"
        justifyContent="space-between"
      >
        <ButtonGroup isAttached>
          <IAIIconButton
            tooltip={t('gallery.images')}
            aria-label={t('gallery.images')}
            onClick={handleClickImagesCategory}
            isChecked={categories === IMAGE_CATEGORIES}
            size="sm"
            icon={<FaImage />}
          />
          <IAIIconButton
            tooltip={t('gallery.assets')}
            aria-label={t('gallery.assets')}
            onClick={handleClickAssetsCategory}
            isChecked={categories === ASSETS_CATEGORIES}
            size="sm"
            icon={<FaServer />}
          />
        </ButtonGroup>
        <Flex gap={2}>
          <IAIPopover
            triggerComponent={
              <IAIIconButton
                tooltip={t('gallery.gallerySettings')}
                aria-label={t('gallery.gallerySettings')}
                size="sm"
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
                  endReached={handleEndReached}
                  scrollerRef={(ref) => setScrollerRef(ref)}
                  itemContent={(index, image) => (
                    <Flex sx={{ pb: 2 }}>
                      <HoverableImage
                        key={`${image.image_name}-${image.thumbnail_url}`}
                        image={image}
                        isSelected={
                          selectedImage?.image_name === image?.image_name
                        }
                      />
                    </Flex>
                  )}
                />
              ) : (
                <VirtuosoGrid
                  style={{ height: '100%' }}
                  data={images}
                  endReached={handleEndReached}
                  components={{
                    Item: ItemContainer,
                    List: ListContainer,
                  }}
                  scrollerRef={setScroller}
                  itemContent={(index, image) => (
                    <HoverableImage
                      key={`${image.image_name}-${image.thumbnail_url}`}
                      image={image}
                      isSelected={
                        selectedImage?.image_name === image?.image_name
                      }
                    />
                  )}
                />
              )}
            </Box>
            <IAIButton
              onClick={handleLoadMoreImages}
              isDisabled={!areMoreImagesAvailable}
              isLoading={isLoading}
              loadingText="Loading"
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
