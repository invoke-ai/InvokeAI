import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Image, Skeleton, Text, useShiftModifier } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { galleryImageClicked } from 'app/store/middleware/listenerMiddleware/listeners/galleryImageClicked';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { useBoolean } from 'common/hooks/useBoolean';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { multipleImageDndSource, singleImageDndSource } from 'features/dnd2/types';
import { useImageContextMenu } from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import { getGalleryImageDataTestId } from 'features/gallery/components/ImageGrid/getGalleryImageDataTestId';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { imageToCompareChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold, PiStarBold, PiStarFill, PiTrashSimpleFill } from 'react-icons/pi';
import { useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

// This class name is used to calculate the number of images that fit in the gallery
export const GALLERY_IMAGE_CONTAINER_CLASS_NAME = 'gallery-image-container';

const galleryImageContainerSX = {
  containerType: 'inline-size',
  w: 'full',
  h: 'full',
  '.gallery-image-size-badge': {
    '@container (max-width: 80px)': {
      '&': { display: 'none' },
    },
  },
  '.gallery-image': {
    touchAction: 'none',
    userSelect: 'none',
    webkitUserSelect: 'none',
    position: 'relative',
    justifyContent: 'center',
    alignItems: 'center',
    aspectRatio: '1/1',
    '::before': {
      content: '""',
      display: 'inline-block',
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      pointerEvents: 'none',
      borderRadius: 'base',
    },
    '&[data-selected=true]::before': {
      boxShadow:
        'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-500), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
    },
    '&[data-selected-for-compare=true]::before': {
      boxShadow:
        'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-300), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
    },
    '&:hover::before': {
      boxShadow:
        'inset 0px 0px 0px 2px var(--invoke-colors-invokeBlue-300), inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-800)',
    },
    '&:hover[data-selected=true]::before': {
      boxShadow:
        'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-400), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
    },
    '&:hover[data-selected-for-compare=true]::before': {
      boxShadow:
        'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-200), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
    },
  },
} satisfies SystemStyleObject;

interface HoverableImageProps {
  imageDTO: ImageDTO;
}

const selectAlwaysShouldImageSizeBadge = createSelector(
  selectGallerySlice,
  (gallery) => gallery.alwaysShowImageSizeBadge
);

export const GalleryImage = memo(({ imageDTO }: HoverableImageProps) => {
  const store = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [element, ref] = useState<HTMLImageElement | null>(null);
  const imageViewer = useImageViewer();
  const selectIsSelectedForCompare = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.imageToCompare?.image_name === imageDTO.image_name),
    [imageDTO.image_name]
  );
  const isSelectedForCompare = useAppSelector(selectIsSelectedForCompare);
  const selectIsSelected = useMemo(
    () =>
      createSelector(selectGallerySlice, (gallery) =>
        gallery.selection.some((i) => i.image_name === imageDTO.image_name)
      ),
    [imageDTO.image_name]
  );
  const isSelected = useAppSelector(selectIsSelected);

  useEffect(() => {
    if (!element) {
      return;
    }
    return combine(
      draggable({
        element,
        getInitialData: () => {
          const { gallery } = store.getState();
          // When we have multiple images selected, and the dragged image is part of the selection, initiate a
          // multi-image drag.
          if (gallery.selection.length > 1 && gallery.selection.includes(imageDTO)) {
            return multipleImageDndSource.getData({ imageDTOs: gallery.selection, boardId: gallery.selectedBoardId });
          }

          // Otherwise, initiate a single-image drag
          return singleImageDndSource.getData({ imageDTO });
        },
        // This is a "local" drag start event, meaning that it is only called when this specific image is dragged.
        onDragStart: (args) => {
          // When we start dragging a single image, set the dragging state to true. This is only called when this
          // specific image is dragged.
          if (singleImageDndSource.typeGuard(args.source.data)) {
            setIsDragging(true);
            return;
          }
        },
      }),
      monitorForElements({
        // This is a "global" drag start event, meaning that it is called for all drag events.
        onDragStart: (args) => {
          // When we start dragging multiple images, set the dragging state to true if the dragged image is part of the
          // selection. This is called for all drag events.
          if (multipleImageDndSource.typeGuard(args.source.data) && args.source.data.imageDTOs.includes(imageDTO)) {
            setIsDragging(true);
          }
        },
        onDrop: () => {
          // Always set the dragging state to false when a drop event occurs.
          setIsDragging(false);
        },
      })
    );
  }, [imageDTO, element, store]);

  const isHovered = useBoolean(false);

  const onClick = useCallback<MouseEventHandler<HTMLDivElement>>(
    (e) => {
      store.dispatch(
        galleryImageClicked({
          imageDTO,
          shiftKey: e.shiftKey,
          ctrlKey: e.ctrlKey,
          metaKey: e.metaKey,
          altKey: e.altKey,
        })
      );
    },
    [imageDTO, store]
  );

  const onDoubleClick = useCallback<MouseEventHandler<HTMLDivElement>>(() => {
    imageViewer.open();
    store.dispatch(imageToCompareChanged(null));
  }, [imageViewer, store]);

  const dataTestId = useMemo(() => getGalleryImageDataTestId(imageDTO.image_name), [imageDTO.image_name]);

  useImageContextMenu(imageDTO, element);

  return (
    <Box
      className={GALLERY_IMAGE_CONTAINER_CLASS_NAME}
      data-testid={dataTestId}
      sx={galleryImageContainerSX}
      opacity={isDragging ? 0.3 : 1}
    >
      <Flex
        role="button"
        className="gallery-image"
        onMouseOver={isHovered.setTrue}
        onMouseOut={isHovered.setFalse}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        data-selected={isSelected}
        data-selected-for-compare={isSelectedForCompare}
      >
        <Image
          ref={ref}
          src={imageDTO.thumbnail_url}
          fallback={<SizedSkeleton width={imageDTO.width} height={imageDTO.height} />}
          w={imageDTO.width}
          objectFit="contain"
          maxW="full"
          maxH="full"
          borderRadius="base"
        />
        <HoverIcons imageDTO={imageDTO} isHovered={isHovered.isTrue} />
      </Flex>
    </Box>
  );
});

GalleryImage.displayName = 'GalleryImage';

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
    if (imageDTO.starred) {
      unstarImages({ imageDTOs: [imageDTO] });
    } else {
      starImages({ imageDTOs: [imageDTO] });
    }
  }, [starImages, unstarImages, imageDTO]);

  const starIcon = useMemo(() => {
    if (imageDTO.starred) {
      return customStarUi ? customStarUi.on.icon : <PiStarFill />;
    } else {
      return customStarUi ? customStarUi.off.icon : <PiStarBold />;
    }
  }, [imageDTO.starred, customStarUi]);

  const starTooltip = useMemo(() => {
    if (imageDTO.starred) {
      return customStarUi ? customStarUi.off.text : 'Unstar';
    } else {
      return customStarUi ? customStarUi.on.text : 'Star';
    }
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
      className="gallery-image-size-badge"
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
      pointerEvents="none"
    >{`${imageDTO.width}x${imageDTO.height}`}</Text>
  );
});

SizeBadge.displayName = 'SizeBadge';

const SizedSkeleton = memo(({ width, height }: { width: number; height: number }) => {
  return <Skeleton w={`${width}px`} h="auto" objectFit="contain" aspectRatio={`${width}/${height}`} />;
});

SizedSkeleton.displayName = 'SizedSkeleton';
