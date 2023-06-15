import { Box, Flex, Icon, Image, MenuItem, MenuList } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useContext, useState } from 'react';
import {
  FaCheck,
  FaExpand,
  FaFolder,
  FaImage,
  FaShare,
  FaTrash,
} from 'react-icons/fa';
import { ContextMenu } from 'chakra-ui-contextmenu';
import {
  resizeAndScaleCanvas,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';
import IAIIconButton from 'common/components/IAIIconButton';
import { ExternalLinkIcon } from '@chakra-ui/icons';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';
import { createSelector } from '@reduxjs/toolkit';
import { systemSelector } from 'features/system/store/systemSelectors';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash-es';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import { sentImageToCanvas, sentImageToImg2Img } from '../store/actions';
import { useAppToaster } from 'app/components/Toaster';
import { ImageDTO } from 'services/api';
import { useDraggable } from '@dnd-kit/core';
import { DeleteImageContext } from 'app/contexts/DeleteImageContext';
import { imageAddedToBoard } from '../../../services/thunks/board';
import { setUpdateBoardModalOpen } from '../store/boardSlice';
import { AddImageToBoardContext } from '../../../app/contexts/AddImageToBoardContext';

export const selector = createSelector(
  [gallerySelector, systemSelector, lightboxSelector, activeTabNameSelector],
  (gallery, system, lightbox, activeTabName) => {
    const {
      galleryImageObjectFit,
      galleryImageMinimumWidth,
      shouldUseSingleGalleryColumn,
    } = gallery;

    const { isLightboxOpen } = lightbox;
    const { isConnected, isProcessing, shouldConfirmOnDelete } = system;

    return {
      canDeleteImage: isConnected && !isProcessing,
      shouldConfirmOnDelete,
      galleryImageObjectFit,
      galleryImageMinimumWidth,
      shouldUseSingleGalleryColumn,
      activeTabName,
      isLightboxOpen,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

interface HoverableImageProps {
  image: ImageDTO;
  isSelected: boolean;
}

const memoEqualityCheck = (
  prev: HoverableImageProps,
  next: HoverableImageProps
) =>
  prev.image.image_name === next.image.image_name &&
  prev.isSelected === next.isSelected;

/**
 * Gallery image component with delete/use all/use seed buttons on hover.
 */
const HoverableImage = memo((props: HoverableImageProps) => {
  const dispatch = useAppDispatch();
  const {
    activeTabName,
    galleryImageObjectFit,
    galleryImageMinimumWidth,
    canDeleteImage,
    shouldUseSingleGalleryColumn,
  } = useAppSelector(selector);

  const { image, isSelected } = props;
  const { image_url, thumbnail_url, image_name } = image;

  const [isHovered, setIsHovered] = useState<boolean>(false);
  const toaster = useAppToaster();

  const { t } = useTranslation();
  const isLightboxEnabled = useFeatureStatus('lightbox').isFeatureEnabled;
  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;

  const { onDelete } = useContext(DeleteImageContext);
  const { onClickAddToBoard } = useContext(AddImageToBoardContext);
  const handleDelete = useCallback(() => {
    onDelete(image);
  }, [image, onDelete]);
  const { recallBothPrompts, recallSeed, recallAllParameters } =
    useRecallParameters();

  const { attributes, listeners, setNodeRef } = useDraggable({
    id: `galleryImage_${image_name}`,
    data: {
      image,
    },
  });

  const handleMouseOver = () => setIsHovered(true);
  const handleMouseOut = () => setIsHovered(false);

  const handleSelectImage = useCallback(() => {
    dispatch(imageSelected(image));
  }, [image, dispatch]);

  // Recall parameters handlers
  const handleRecallPrompt = useCallback(() => {
    recallBothPrompts(
      image.metadata?.positive_conditioning,
      image.metadata?.negative_conditioning
    );
  }, [
    image.metadata?.negative_conditioning,
    image.metadata?.positive_conditioning,
    recallBothPrompts,
  ]);

  const handleRecallSeed = useCallback(() => {
    recallSeed(image.metadata?.seed);
  }, [image, recallSeed]);

  const handleSendToImageToImage = useCallback(() => {
    dispatch(sentImageToImg2Img());
    dispatch(initialImageSelected(image));
  }, [dispatch, image]);

  // const handleRecallInitialImage = useCallback(() => {
  //   recallInitialImage(image.metadata.invokeai?.node?.image);
  // }, [image, recallInitialImage]);

  /**
   * TODO: the rest of these
   */
  const handleSendToCanvas = () => {
    dispatch(sentImageToCanvas());
    dispatch(setInitialCanvasImage(image));

    dispatch(resizeAndScaleCanvas());

    if (activeTabName !== 'unifiedCanvas') {
      dispatch(setActiveTab('unifiedCanvas'));
    }

    toaster({
      title: t('toast.sentToUnifiedCanvas'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseAllParameters = useCallback(() => {
    recallAllParameters(image);
  }, [image, recallAllParameters]);

  const handleLightBox = () => {
    // dispatch(setCurrentImage(image));
    // dispatch(setIsLightboxOpen(true));
  };

  const handleAddToBoard = useCallback(() => {
    onClickAddToBoard(image);
  }, [image, onClickAddToBoard]);

  const handleOpenInNewTab = () => {
    window.open(image.image_url, '_blank');
  };

  return (
    <Box
      ref={setNodeRef}
      {...listeners}
      {...attributes}
      sx={{ w: 'full', h: 'full', touchAction: 'none' }}
    >
      <ContextMenu<HTMLDivElement>
        menuProps={{ size: 'sm', isLazy: true }}
        renderMenu={() => (
          <MenuList sx={{ visibility: 'visible !important' }}>
            <MenuItem
              icon={<ExternalLinkIcon />}
              onClickCapture={handleOpenInNewTab}
            >
              {t('common.openInNewTab')}
            </MenuItem>
            {isLightboxEnabled && (
              <MenuItem icon={<FaExpand />} onClickCapture={handleLightBox}>
                {t('parameters.openInViewer')}
              </MenuItem>
            )}
            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleRecallPrompt}
              isDisabled={image?.metadata?.positive_conditioning === undefined}
            >
              {t('parameters.usePrompt')}
            </MenuItem>

            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleRecallSeed}
              isDisabled={image?.metadata?.seed === undefined}
            >
              {t('parameters.useSeed')}
            </MenuItem>
            {/* <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleRecallInitialImage}
              isDisabled={image?.metadata?.type !== 'img2img'}
            >
              {t('parameters.useInitImg')}
            </MenuItem> */}
            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleUseAllParameters}
              isDisabled={
                // what should these be
                !['t2l', 'l2l', 'inpaint'].includes(
                  String(image?.metadata?.type)
                )
              }
            >
              {t('parameters.useAll')}
            </MenuItem>
            <MenuItem
              icon={<FaShare />}
              onClickCapture={handleSendToImageToImage}
              id="send-to-img2img"
            >
              {t('parameters.sendToImg2Img')}
            </MenuItem>
            {isCanvasEnabled && (
              <MenuItem
                icon={<FaShare />}
                onClickCapture={handleSendToCanvas}
                id="send-to-canvas"
              >
                {t('parameters.sendToUnifiedCanvas')}
              </MenuItem>
            )}
            <MenuItem icon={<FaFolder />} onClickCapture={handleAddToBoard}>
              Add to Board
            </MenuItem>
            <MenuItem
              sx={{ color: 'error.300' }}
              icon={<FaTrash />}
              onClickCapture={handleDelete}
            >
              {t('gallery.deleteImage')}
            </MenuItem>
          </MenuList>
        )}
      >
        {(ref) => (
          <Box
            position="relative"
            key={image_name}
            onMouseOver={handleMouseOver}
            onMouseOut={handleMouseOut}
            userSelect="none"
            onClick={handleSelectImage}
            ref={ref}
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              w: 'full',
              h: 'full',
              transition: 'transform 0.2s ease-out',
              aspectRatio: '1/1',
              cursor: 'pointer',
            }}
          >
            <Image
              loading="lazy"
              objectFit={
                shouldUseSingleGalleryColumn ? 'contain' : galleryImageObjectFit
              }
              draggable={false}
              rounded="md"
              src={thumbnail_url || image_url}
              fallback={<FaImage />}
              sx={{
                width: '100%',
                height: '100%',
                maxWidth: '100%',
                maxHeight: '100%',
              }}
            />
            {isSelected && (
              <Flex
                sx={{
                  position: 'absolute',
                  top: '0',
                  insetInlineStart: '0',
                  width: '100%',
                  height: '100%',
                  alignItems: 'center',
                  justifyContent: 'center',
                  pointerEvents: 'none',
                }}
              >
                <Icon
                  filter={'drop-shadow(0px 0px 1rem black)'}
                  as={FaCheck}
                  sx={{
                    width: '50%',
                    height: '50%',
                    maxWidth: '4rem',
                    maxHeight: '4rem',
                    fill: 'ok.500',
                  }}
                />
              </Flex>
            )}
            {isHovered && galleryImageMinimumWidth >= 100 && (
              <Box
                sx={{
                  position: 'absolute',
                  top: 1,
                  insetInlineEnd: 1,
                }}
              >
                <IAIIconButton
                  onClickCapture={handleDelete}
                  aria-label={t('gallery.deleteImage')}
                  icon={<FaTrash />}
                  size="xs"
                  fontSize={14}
                  isDisabled={!canDeleteImage}
                />
              </Box>
            )}
          </Box>
        )}
      </ContextMenu>
    </Box>
  );
}, memoEqualityCheck);

HoverableImage.displayName = 'HoverableImage';

export default HoverableImage;
