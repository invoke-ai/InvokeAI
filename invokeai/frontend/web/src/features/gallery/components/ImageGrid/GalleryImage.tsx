import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Text, useShiftModifier } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import IAIFillSkeleton from 'common/components/IAIFillSkeleton';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import type { GallerySelectionDraggableData, ImageDraggableData, TypesafeDraggableData } from 'features/dnd/types';
import { getGalleryImageDataTestId } from 'features/gallery/components/ImageGrid/getGalleryImageDataTestId';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useMultiselect } from 'features/gallery/hooks/useMultiselect';
import { useScrollIntoView } from 'features/gallery/hooks/useScrollIntoView';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import type { MouseEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold, PiStarBold, PiStarFill, PiTrashSimpleFill } from 'react-icons/pi';
import { useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

// This class name is used to calculate the number of images that fit in the gallery
export const GALLERY_IMAGE_CLASS_NAME = 'gallery-image';

const imageSx: SystemStyleObject = { w: 'full', h: 'full' };
const boxSx: SystemStyleObject = {
  containerType: 'inline-size',
};

const badgeSx: SystemStyleObject = {
  '@container (max-width: 80px)': {
    '&': { display: 'none' },
  },
};

interface HoverableImageProps {
  imageDTO: ImageDTO;
  index: number;
}

const selectAlwaysShouldImageSizeBadge = createSelector(
  selectGallerySlice,
  (gallery) => gallery.alwaysShowImageSizeBadge
);

export const GalleryImage = memo(({ index, imageDTO }: HoverableImageProps) => {
  if (!imageDTO) {
    return <IAIFillSkeleton />;
  }

  return <GalleryImageContent index={index} imageDTO={imageDTO} />;
});

GalleryImage.displayName = 'GalleryImage';

const GalleryImageContent = memo(({ index, imageDTO }: HoverableImageProps) => {
  const dispatch = useAppDispatch();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const selectIsSelectedForCompare = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.imageToCompare?.image_name === imageDTO.image_name),
    [imageDTO.image_name]
  );
  const isSelectedForCompare = useAppSelector(selectIsSelectedForCompare);
  const { handleClick, isSelected, areMultiplesSelected } = useMultiselect(imageDTO);

  const imageContainerRef = useScrollIntoView(isSelected, index, areMultiplesSelected);

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (areMultiplesSelected) {
      const data: GallerySelectionDraggableData = {
        id: 'gallery-image',
        payloadType: 'GALLERY_SELECTION',
        payload: { boardId: selectedBoardId },
      };
      return data;
    }

    if (imageDTO) {
      const data: ImageDraggableData = {
        id: 'gallery-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
      return data;
    }
  }, [imageDTO, selectedBoardId, areMultiplesSelected]);

  const [isHovered, setIsHovered] = useState(false);

  const handleMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const imageViewer = useImageViewer();
  const onDoubleClick = useCallback(() => {
    imageViewer.open();
    dispatch(imageToCompareChanged(null));
  }, [dispatch, imageViewer]);

  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const dataTestId = useMemo(() => getGalleryImageDataTestId(imageDTO.image_name), [imageDTO.image_name]);

  if (!imageDTO) {
    return <IAIFillSkeleton />;
  }

  return (
    <Box w="full" h="full" className={GALLERY_IMAGE_CLASS_NAME} data-testid={dataTestId} sx={boxSx}>
      <Flex
        ref={imageContainerRef}
        userSelect="none"
        position="relative"
        justifyContent="center"
        alignItems="center"
        aspectRatio="1/1"
        onMouseOver={handleMouseOver}
        onMouseOut={handleMouseOut}
      >
        <IAIDndImage
          onClick={handleClick}
          onDoubleClick={onDoubleClick}
          imageDTO={imageDTO}
          draggableData={draggableData}
          isSelected={isSelected}
          isSelectedForCompare={isSelectedForCompare}
          minSize={0}
          imageSx={imageSx}
          isDropDisabled={true}
          isUploadDisabled={true}
          thumbnail={true}
          withHoverOverlay
        >
          <HoverIcons imageDTO={imageDTO} isHovered={isHovered} />
        </IAIDndImage>
      </Flex>
    </Box>
  );
});

GalleryImageContent.displayName = 'GalleryImageContent';

const HoverIcons = memo(({ imageDTO, isHovered }: { imageDTO: ImageDTO; isHovered: boolean }) => {
  const alwaysShowImageSizeBadge = useAppSelector(selectAlwaysShouldImageSizeBadge);

  return (
    <>
      {(isHovered || alwaysShowImageSizeBadge) && <SizeBadge imageDTO={imageDTO} />}
      {(isHovered || imageDTO.starred) && <StarIcon imageDTO={imageDTO} />}
      {isHovered && <DeleteIcon imageDTO={imageDTO} />}
      {isHovered && <OpenInViewerIconButton imageDTO={imageDTO} />}
    </>
  );
});
HoverIcons.displayName = 'HoverIcons';

const DeleteIcon = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  const shift = useShiftModifier();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      if (!imageDTO) {
        return;
      }
      dispatch(imagesToDeleteSelected([imageDTO]));
    },
    [dispatch, imageDTO]
  );

  if (!shift) {
    return null;
  }

  return (
    <IAIDndImageIcon
      onClick={onClick}
      icon={<PiTrashSimpleFill />}
      tooltip={t('gallery.deleteImage_one')}
      position="absolute"
      bottom={2}
      insetInlineEnd={2}
    />
  );
});

DeleteIcon.displayName = 'DeleteIcon';

const OpenInViewerIconButton = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  const imageViewer = useImageViewer();
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    imageViewer.openImageInViewer(imageDTO);
  }, [imageDTO, imageViewer]);

  return (
    <IAIDndImageIcon
      onClick={onClick}
      icon={<PiArrowsOutBold />}
      tooltip={t('gallery.openInViewer')}
      position="absolute"
      insetBlockStart={2}
      insetInlineStart={2}
    />
  );
});

OpenInViewerIconButton.displayName = 'OpenInViewerIconButton';

const StarIcon = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  const customStarUi = useStore($customStarUI);
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const toggleStarredState = useCallback(() => {
    if (imageDTO) {
      if (imageDTO.starred) {
        unstarImages({ imageDTOs: [imageDTO] });
      }
      if (!imageDTO.starred) {
        starImages({ imageDTOs: [imageDTO] });
      }
    }
  }, [starImages, unstarImages, imageDTO]);

  const starIcon = useMemo(() => {
    if (imageDTO.starred) {
      return customStarUi ? customStarUi.on.icon : <PiStarFill />;
    }
    if (!imageDTO.starred) {
      return customStarUi ? customStarUi.off.icon : <PiStarBold />;
    }
  }, [imageDTO.starred, customStarUi]);

  const starTooltip = useMemo(() => {
    if (imageDTO.starred) {
      return customStarUi ? customStarUi.off.text : 'Unstar';
    }
    if (!imageDTO.starred) {
      return customStarUi ? customStarUi.on.text : 'Star';
    }
    return '';
  }, [imageDTO.starred, customStarUi]);

  return (
    <IAIDndImageIcon
      onClick={toggleStarredState}
      icon={starIcon}
      tooltip={starTooltip}
      position="absolute"
      top={2}
      insetInlineEnd={2}
    />
  );
});

StarIcon.displayName = 'StarIcon';

const SizeBadge = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  return (
    <Text
      position="absolute"
      background="base.900"
      color="base.50"
      fontSize="sm"
      fontWeight="semibold"
      bottom={1}
      left={1}
      opacity={0.7}
      px={2}
      lineHeight={1.25}
      borderTopEndRadius="base"
      sx={badgeSx}
      pointerEvents="none"
    >{`${imageDTO.width}x${imageDTO.height}`}</Text>
  );
});

SizeBadge.displayName = 'SizeBadge';
