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
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStarBold, PiStarFill, PiTrashSimpleFill } from 'react-icons/pi';
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

const GalleryImage = ({ index, imageDTO }: HoverableImageProps) => {
  const dispatch = useAppDispatch();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const selectIsSelectedForCompare = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.imageToCompare?.image_name === imageDTO.image_name),
    [imageDTO.image_name]
  );
  const alwaysShowImageSizeBadge = useAppSelector(selectAlwaysShouldImageSizeBadge);
  const isSelectedForCompare = useAppSelector(selectIsSelectedForCompare);
  const { handleClick, isSelected, areMultiplesSelected } = useMultiselect(imageDTO);

  const customStarUi = useStore($customStarUI);

  const imageContainerRef = useScrollIntoView(isSelected, index, areMultiplesSelected);

  const handleDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      if (!imageDTO) {
        return;
      }
      dispatch(imagesToDeleteSelected([imageDTO]));
    },
    [dispatch, imageDTO]
  );

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

  const starIcon = useMemo(() => {
    if (imageDTO.starred) {
      return customStarUi ? customStarUi.on.icon : <PiStarFill size="20" />;
    }
    if (!imageDTO.starred && isHovered) {
      return customStarUi ? customStarUi.off.icon : <PiStarBold size="20" />;
    }
  }, [imageDTO.starred, isHovered, customStarUi]);

  const starTooltip = useMemo(() => {
    if (imageDTO.starred) {
      return customStarUi ? customStarUi.off.text : 'Unstar';
    }
    if (!imageDTO.starred) {
      return customStarUi ? customStarUi.on.text : 'Star';
    }
    return '';
  }, [imageDTO.starred, customStarUi]);

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
          onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}
        >
          <>
            {(isHovered || alwaysShowImageSizeBadge) && (
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
            )}
            <IAIDndImageIcon
              onClick={toggleStarredState}
              icon={starIcon}
              tooltip={starTooltip}
              position="absolute"
              top={2}
              insetInlineEnd={2}
            />

            {isHovered && <DeleteIcon onClick={handleDelete} />}
          </>
        </IAIDndImage>
      </Flex>
    </Box>
  );
};

export default memo(GalleryImage);

const DeleteIcon = ({ onClick }: { onClick: MouseEventHandler }) => {
  const shift = useShiftModifier();
  const { t } = useTranslation();

  if (!shift) {
    return null;
  }

  return (
    <IAIDndImageIcon
      onClick={onClick}
      icon={<PiTrashSimpleFill size="16px" />}
      tooltip={t('gallery.deleteImage_one')}
      position="absolute"
      bottom={2}
      insetInlineEnd={2}
    />
  );
};
