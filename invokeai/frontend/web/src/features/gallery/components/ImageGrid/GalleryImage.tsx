import type { SystemStyleObject } from '@chakra-ui/react';
import { Box, Flex } from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import IAIFillSkeleton from 'common/components/IAIFillSkeleton';
import { $shift } from 'common/hooks/useGlobalModifiers';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import type {
  ImageDraggableData,
  ImageDTOsDraggableData,
  TypesafeDraggableData,
} from 'features/dnd/types';
import type { VirtuosoGalleryContext } from 'features/gallery/components/ImageGrid/types';
import { useMultiselect } from 'features/gallery/hooks/useMultiselect';
import { useScrollToVisible } from 'features/gallery/hooks/useScrollToVisible';
import type { MouseEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStarBold, PiStarFill, PiTrashSimpleFill } from 'react-icons/pi'
import {
  useGetImageDTOQuery,
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';

const imageSx: SystemStyleObject = { w: 'full', h: 'full' };
const imageIconStyleOverrides: SystemStyleObject = {
  bottom: 2,
  top: 'auto',
};
interface HoverableImageProps {
  imageName: string;
  index: number;
  virtuosoContext: VirtuosoGalleryContext;
}

const GalleryImage = (props: HoverableImageProps) => {
  const dispatch = useAppDispatch();
  const { imageName, virtuosoContext } = props;
  const { currentData: imageDTO } = useGetImageDTOQuery(imageName);
  const shift = useStore($shift);
  const { t } = useTranslation();

  const { handleClick, isSelected, selection, selectionCount } =
    useMultiselect(imageDTO);

  const customStarUi = useStore($customStarUI);

  const imageContainerRef = useScrollToVisible(
    isSelected,
    props.index,
    selectionCount,
    virtuosoContext
  );

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
    if (selectionCount > 1) {
      const data: ImageDTOsDraggableData = {
        id: 'gallery-image',
        payloadType: 'IMAGE_DTOS',
        payload: { imageDTOs: selection },
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
  }, [imageDTO, selection, selectionCount]);

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

  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const starIcon = useMemo(() => {
    if (imageDTO?.starred) {
      return customStarUi ? customStarUi.on.icon : <PiStarFill size="20" />;
    }
    if (!imageDTO?.starred && isHovered) {
      return customStarUi ? customStarUi.off.icon : <PiStarBold size="20" />;
    }
  }, [imageDTO?.starred, isHovered, customStarUi]);

  const starTooltip = useMemo(() => {
    if (imageDTO?.starred) {
      return customStarUi ? customStarUi.off.text : 'Unstar';
    }
    if (!imageDTO?.starred) {
      return customStarUi ? customStarUi.on.text : 'Star';
    }
    return '';
  }, [imageDTO?.starred, customStarUi]);

  if (!imageDTO) {
    return <IAIFillSkeleton />;
  }

  return (
    <Box w="full" h="full" data-testid={`image-${imageDTO.image_name}`}>
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
          imageDTO={imageDTO}
          draggableData={draggableData}
          isSelected={isSelected}
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
            <IAIDndImageIcon
              onClick={toggleStarredState}
              icon={starIcon}
              tooltip={starTooltip}
            />

            {isHovered && shift && (
              <IAIDndImageIcon
                onClick={handleDelete}
                icon={<PiTrashSimpleFill size="16px" />}
                tooltip={t('gallery.deleteImage')}
                styleOverrides={imageIconStyleOverrides}
              />
            )}
          </>
        </IAIDndImage>
      </Flex>
    </Box>
  );
};

export default memo(GalleryImage);
