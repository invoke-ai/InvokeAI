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
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import IAISlider from 'common/components/IAISlider';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import {
  setGalleryImageMinimumWidth,
  setGalleryImageObjectFit,
  setShouldAutoSwitchToNewImages,
  setShouldUseSingleGalleryColumn,
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
import { boardSelector } from '../store/boardSelectors';
import { boardCreated } from '../../../services/thunks/board';
import BoardsList from './Boards/BoardsList';
import { selectBoardsById } from '../store/boardSlice';

const itemSelector = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { images, boards } = state;

    const { categories } = images;

    const allImages = selectImagesAll(state);
    const items = allImages.filter((i) =>
      categories.includes(i.image_category)
    );
    const areMoreAvailable = items.length < images.total;
    const isLoading = images.isLoading;

    const selectedBoard = boards.selectedBoardId
      ? selectBoardsById(state, boards.selectedBoardId)
      : undefined;

    return {
      items,
      isLoading,
      areMoreAvailable,
      categories: images.categories,
      selectedBoard,
    };
  },
  defaultSelectorOptions
);

const mainSelector = createSelector(
  [gallerySelector, uiSelector, boardSelector],
  (gallery, ui, boardState) => {
    const {
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      shouldAutoSwitchToNewImages,
      shouldUseSingleGalleryColumn,
      selectedImage,
      galleryView,
    } = gallery;

    const { shouldPinGallery } = ui;

    const { entities: boards } = boardState;

    return {
      shouldPinGallery,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      shouldAutoSwitchToNewImages,
      shouldUseSingleGalleryColumn,
      selectedImage,
      galleryView,
      boards,
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
    galleryView,
    boards,
  } = useAppSelector(mainSelector);

  const { items, areMoreAvailable, isLoading, categories, selectedBoard } =
    useAppSelector(itemSelector);

  const handleLoadMoreImages = useCallback(() => {
    dispatch(receivedPageOfImages());
  }, [dispatch]);

  const handleEndReached = useMemo(() => {
    if (areMoreAvailable && !isLoading) {
      return handleLoadMoreImages;
    }
    return undefined;
  }, [areMoreAvailable, handleLoadMoreImages, isLoading]);

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
    dispatch(setGalleryView('images'));
  }, [dispatch]);

  const handleClickAssetsCategory = useCallback(() => {
    dispatch(imageCategoriesChanged(ASSETS_CATEGORIES));
    dispatch(setGalleryView('assets'));
  }, [dispatch]);

  const handleClickBoardsView = useCallback(() => {
    dispatch(setGalleryView('boards'));
  }, [dispatch]);

  const [newBoardName, setNewBoardName] = useState('');

  const handleCreateNewBoard = () => {
    dispatch(boardCreated({ requestBody: newBoardName }));
  };

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
        {selectedBoard && (
          <Flex>
            <Text>{selectedBoard.board_name}</Text>
          </Flex>
        )}
        <Flex gap={2}>
          {/* <IAIPopover
            triggerComponent={
              <IAIIconButton
                tooltip="Add Board"
                aria-label="Add Board"
                size="sm"
                icon={<FaPlus />}
              />
            }
          >
            <Flex direction="column" gap={2}>
              <IAIInput
                label="Board Name"
                placeholder="Board Name"
                value={newBoardName}
                onChange={(e) => setNewBoardName(e.target.value)}
              />
              <IAIButton
                size="sm"
                onClick={handleCreateNewBoard}
                disabled={true}
                isLoading={false}
              >
                Create
              </IAIButton>
            </Flex>
          </IAIPopover> */}
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
              <IAISimpleCheckbox
                label={t('gallery.autoSwitchNewImages')}
                isChecked={shouldAutoSwitchToNewImages}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  dispatch(setShouldAutoSwitchToNewImages(e.target.checked))
                }
              />
              <IAISimpleCheckbox
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
      <Box>
        <BoardsList />
      </Box>
      <Flex direction="column" gap={2} h="full">
        {items.length || areMoreAvailable ? (
          <>
            <Box ref={rootRef} data-overlayscrollbars="" h="100%">
              {shouldUseSingleGalleryColumn ? (
                <Virtuoso
                  style={{ height: '100%' }}
                  data={items}
                  endReached={handleEndReached}
                  scrollerRef={(ref) => setScrollerRef(ref)}
                  itemContent={(index, item) => (
                    <Flex sx={{ pb: 2 }}>
                      <HoverableImage
                        key={`${item.image_name}-${item.thumbnail_url}`}
                        image={item}
                        isSelected={
                          selectedImage?.image_name === item?.image_name
                        }
                      />
                    </Flex>
                  )}
                />
              ) : (
                <VirtuosoGrid
                  style={{ height: '100%' }}
                  data={items}
                  endReached={handleEndReached}
                  components={{
                    Item: ItemContainer,
                    List: ListContainer,
                  }}
                  scrollerRef={setScroller}
                  itemContent={(index, item) => (
                    <HoverableImage
                      key={`${item.image_name}-${item.thumbnail_url}`}
                      image={item}
                      isSelected={
                        selectedImage?.image_name === item?.image_name
                      }
                    />
                  )}
                />
              )}
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
