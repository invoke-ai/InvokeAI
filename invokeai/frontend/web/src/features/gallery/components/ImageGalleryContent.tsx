import {
  Box,
  Button,
  ButtonGroup,
  Flex,
  FlexProps,
  Grid,
  Skeleton,
  Text,
  VStack,
  forwardRef,
  useColorMode,
  useDisclosure,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import IAISlider from 'common/components/IAISlider';
import {
  setGalleryImageMinimumWidth,
  setGalleryView,
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
import GalleryImage from './GalleryImage';

import { ChevronUpIcon } from '@chakra-ui/icons';
import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  imageCategoriesChanged,
  selectFilteredImages,
  shouldAutoSwitchChanged,
} from 'features/gallery/store/gallerySlice';
import { VirtuosoGrid } from 'react-virtuoso';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { receivedPageOfImages } from 'services/api/thunks/image';
import { ImageDTO } from 'services/api/types';
import { mode } from 'theme/util/mode';
import BoardsList from './Boards/BoardsList';

const LOADING_IMAGE_ARRAY = Array(20).fill('loading');

const selector = createSelector(
  [stateSelector, selectFilteredImages],
  (state, filteredImages) => {
    const {
      categories,
      total: allImagesTotal,
      isLoading,
      selectedBoardId,
      galleryImageMinimumWidth,
      galleryView,
      shouldAutoSwitch,
    } = state.gallery;
    const { shouldPinGallery } = state.ui;

    const images = filteredImages as (ImageDTO | string)[];

    return {
      images: isLoading ? images.concat(LOADING_IMAGE_ARRAY) : images,
      allImagesTotal,
      isLoading,
      categories,
      selectedBoardId,
      shouldPinGallery,
      galleryImageMinimumWidth,
      shouldAutoSwitch,
      galleryView,
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

  const { colorMode } = useColorMode();

  const {
    images,
    isLoading,
    allImagesTotal,
    categories,
    selectedBoardId,
    shouldPinGallery,
    galleryImageMinimumWidth,
    shouldAutoSwitch,
    galleryView,
  } = useAppSelector(selector);

  const { selectedBoard } = useListAllBoardsQuery(undefined, {
    selectFromResult: ({ data }) => ({
      selectedBoard: data?.find((b) => b.board_id === selectedBoardId),
    }),
  });

  const filteredImagesTotal = useMemo(
    () => selectedBoard?.image_count ?? allImagesTotal,
    [allImagesTotal, selectedBoard?.image_count]
  );

  const areMoreAvailable = useMemo(() => {
    return images.length < filteredImagesTotal;
  }, [images.length, filteredImagesTotal]);

  const handleLoadMoreImages = useCallback(() => {
    dispatch(
      receivedPageOfImages({
        categories,
        board_id: selectedBoardId,
        is_intermediate: false,
      })
    );
  }, [categories, dispatch, selectedBoardId]);

  const handleEndReached = useMemo(() => {
    if (areMoreAvailable && !isLoading) {
      return handleLoadMoreImages;
    }
    return undefined;
  }, [areMoreAvailable, handleLoadMoreImages, isLoading]);

  const { isOpen: isBoardListOpen, onToggle } = useDisclosure();

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

  const handleClickImagesCategory = useCallback(() => {
    dispatch(imageCategoriesChanged(IMAGE_CATEGORIES));
    dispatch(setGalleryView('images'));
  }, [dispatch]);

  const handleClickAssetsCategory = useCallback(() => {
    dispatch(imageCategoriesChanged(ASSETS_CATEGORIES));
    dispatch(setGalleryView('assets'));
  }, [dispatch]);

  return (
    <VStack
      sx={{
        flexDirection: 'column',
        h: 'full',
        w: 'full',
        borderRadius: 'base',
      }}
    >
      <Box sx={{ w: 'full' }}>
        <Flex
          ref={resizeObserverRef}
          sx={{
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 2,
          }}
        >
          <ButtonGroup isAttached>
            <IAIIconButton
              tooltip={t('gallery.images')}
              aria-label={t('gallery.images')}
              onClick={handleClickImagesCategory}
              isChecked={galleryView === 'images'}
              size="sm"
              icon={<FaImage />}
            />
            <IAIIconButton
              tooltip={t('gallery.assets')}
              aria-label={t('gallery.assets')}
              onClick={handleClickAssetsCategory}
              isChecked={galleryView === 'assets'}
              size="sm"
              icon={<FaServer />}
            />
          </ButtonGroup>
          <Flex
            as={Button}
            onClick={onToggle}
            size="sm"
            variant="ghost"
            sx={{
              w: 'full',
              justifyContent: 'center',
              alignItems: 'center',
              px: 2,
              _hover: {
                bg: mode('base.100', 'base.800')(colorMode),
              },
            }}
          >
            <Text
              noOfLines={1}
              sx={{
                w: 'full',
                color: mode('base.800', 'base.200')(colorMode),
                fontWeight: 600,
              }}
            >
              {selectedBoard ? selectedBoard.board_name : 'All Images'}
            </Text>
            <ChevronUpIcon
              sx={{
                transform: isBoardListOpen ? 'rotate(0deg)' : 'rotate(180deg)',
                transitionProperty: 'common',
                transitionDuration: 'normal',
              }}
            />
          </Flex>
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
              <IAISimpleCheckbox
                label={t('gallery.autoSwitchNewImages')}
                isChecked={shouldAutoSwitch}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  dispatch(shouldAutoSwitchChanged(e.target.checked))
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
        <Box>
          <BoardsList isOpen={isBoardListOpen} />
        </Box>
      </Box>
      <Flex direction="column" gap={2} h="full" w="full">
        {images.length || areMoreAvailable ? (
          <>
            <Box ref={rootRef} data-overlayscrollbars="" h="100%">
              <VirtuosoGrid
                style={{ height: '100%' }}
                data={images}
                endReached={handleEndReached}
                components={{
                  Item: ItemContainer,
                  List: ListContainer,
                }}
                scrollerRef={setScroller}
                itemContent={(index, item) =>
                  typeof item === 'string' ? (
                    <Skeleton
                      sx={{ w: 'full', h: 'full', aspectRatio: '1/1' }}
                    />
                  ) : (
                    <GalleryImage
                      key={`${item.image_name}-${item.thumbnail_url}`}
                      imageDTO={item}
                    />
                  )
                }
              />
            </Box>
            <IAIButton
              onClick={handleLoadMoreImages}
              isDisabled={!areMoreAvailable}
              isLoading={isLoading}
              loadingText="Loading"
              flexShrink={0}
            >
              {areMoreAvailable
                ? t('gallery.loadMore')
                : t('gallery.allImagesLoaded')}
            </IAIButton>
          </>
        ) : (
          <IAINoContentFallback
            label={t('gallery.noImagesInGallery')}
            icon={FaImage}
          />
        )}
      </Flex>
    </VStack>
  );
};

type ItemContainerProps = PropsWithChildren & FlexProps;
const ItemContainer = forwardRef((props: ItemContainerProps, ref) => (
  <Box className="item-container" ref={ref} p={1.5}>
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
        gridTemplateColumns: `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr));`,
      }}
    >
      {props.children}
    </Grid>
  );
});

export default memo(ImageGalleryContent);
