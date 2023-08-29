import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIFillSkeleton from 'common/components/IAIFillSkeleton';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import {
  ImageDTOsDraggableData,
  ImageDraggableData,
  TypesafeDraggableData,
} from 'features/dnd/types';
import { useMultiselect } from 'features/gallery/hooks/useMultiselect';
import { MouseEvent, memo, useCallback, useMemo, useState } from 'react';
import { FaTrash } from 'react-icons/fa';
import { MdStar, MdStarBorder } from 'react-icons/md';
import {
  useGetImageDTOQuery,
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';
import IAIDndImageIcon from '../../../../common/components/IAIDndImageIcon';

interface HoverableImageProps {
  imageName: string;
}

const GalleryImage = (props: HoverableImageProps) => {
  const dispatch = useAppDispatch();
  const { imageName } = props;
  const { currentData: imageDTO } = useGetImageDTOQuery(imageName);
  const shift = useAppSelector((state) => state.hotkeys.shift);

  const { handleClick, isSelected, selection, selectionCount } =
    useMultiselect(imageDTO);

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
      return <MdStar size="20" />;
    }
    if (!imageDTO?.starred && isHovered) {
      return <MdStarBorder size="20" />;
    }
  }, [imageDTO?.starred, isHovered]);

  if (!imageDTO) {
    return <IAIFillSkeleton />;
  }

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none' }}>
      <Flex
        userSelect="none"
        sx={{
          position: 'relative',
          justifyContent: 'center',
          alignItems: 'center',
          aspectRatio: '1/1',
        }}
      >
        <IAIDndImage
          onClick={handleClick}
          imageDTO={imageDTO}
          draggableData={draggableData}
          isSelected={isSelected}
          minSize={0}
          imageSx={{ w: 'full', h: 'full' }}
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
              tooltip={imageDTO.starred ? 'Unstar' : 'Star'}
            />

            {isHovered && shift && (
              <IAIDndImageIcon
                onClick={handleDelete}
                icon={<FaTrash />}
                tooltip="Delete"
                styleOverrides={{
                  bottom: 2,
                  top: 'auto',
                }}
              />
            )}
          </>
        </IAIDndImage>
      </Flex>
    </Box>
  );
};

export default memo(GalleryImage);
