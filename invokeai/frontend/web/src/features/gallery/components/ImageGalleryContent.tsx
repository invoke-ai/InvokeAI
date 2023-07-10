import {
  Box,
  Button,
  ButtonGroup,
  Flex,
  Text,
  VStack,
  useColorMode,
  useDisclosure,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import IAISlider from 'common/components/IAISlider';
import {
  setGalleryImageMinimumWidth,
  setGalleryView,
} from 'features/gallery/store/gallerySlice';
import { togglePinGalleryPanel } from 'features/ui/store/uiSlice';

import { ChangeEvent, memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
import { FaImage, FaServer, FaWrench } from 'react-icons/fa';

import { ChevronUpIcon } from '@chakra-ui/icons';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  imageCategoriesChanged,
  shouldAutoSwitchChanged,
} from 'features/gallery/store/gallerySlice';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { mode } from 'theme/util/mode';
import BatchGrid from './BatchGrid';
import BoardsList from './Boards/BoardsList';
import ImageGalleryGrid from './ImageGalleryGrid';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const {
      selectedBoardId,
      galleryImageMinimumWidth,
      galleryView,
      shouldAutoSwitch,
    } = state.gallery;
    const { shouldPinGallery } = state.ui;

    return {
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

  const { colorMode } = useColorMode();

  const {
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

  const boardTitle = useMemo(() => {
    if (selectedBoardId === 'batch') {
      return 'Batch';
    }
    if (selectedBoard) {
      return selectedBoard.board_name;
    }
    return 'All Images';
  }, [selectedBoard, selectedBoardId]);

  const { isOpen: isBoardListOpen, onToggle } = useDisclosure();

  const handleChangeGalleryImageMinimumWidth = (v: number) => {
    dispatch(setGalleryImageMinimumWidth(v));
  };

  const handleSetShouldPinGallery = () => {
    dispatch(togglePinGalleryPanel());
    dispatch(requestCanvasRescale());
  };

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
              {boardTitle}
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
        {selectedBoardId === 'batch' ? <BatchGrid /> : <ImageGalleryGrid />}
      </Flex>
    </VStack>
  );
};

export default memo(ImageGalleryContent);
