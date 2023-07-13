import { Box, ChakraProps, Flex, Heading, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { memo } from 'react';
import { TypesafeDraggableData } from './typesafeDnd';

type OverlayDragImageProps = {
  dragData: TypesafeDraggableData | null;
};

const BOX_SIZE = 28;

const STYLES: ChakraProps['sx'] = {
  w: BOX_SIZE,
  h: BOX_SIZE,
  maxW: BOX_SIZE,
  maxH: BOX_SIZE,
  shadow: 'dark-lg',
  borderRadius: 'lg',
  borderWidth: 2,
  borderStyle: 'dashed',
  borderColor: 'base.100',
  opacity: 0.5,
  bg: 'base.800',
  color: 'base.50',
  _dark: {
    borderColor: 'base.200',
    bg: 'base.900',
    color: 'base.100',
  },
};

const selector = createSelector(
  stateSelector,
  (state) => {
    const gallerySelectionCount = state.gallery.selection.length;
    const batchSelectionCount = state.batch.selection.length;

    return {
      gallerySelectionCount,
      batchSelectionCount,
    };
  },
  defaultSelectorOptions
);

const DragPreview = (props: OverlayDragImageProps) => {
  const { gallerySelectionCount, batchSelectionCount } =
    useAppSelector(selector);

  if (!props.dragData) {
    return;
  }

  if (props.dragData.payloadType === 'IMAGE_DTO') {
    const { thumbnail_url, width, height } = props.dragData.payload.imageDTO;
    return (
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          userSelect: 'none',
          cursor: 'none',
        }}
      >
        <Image
          sx={{
            ...STYLES,
          }}
          objectFit="contain"
          src={thumbnail_url}
          width={width}
          height={height}
        />
      </Box>
    );
  }

  if (props.dragData.payloadType === 'BATCH_SELECTION') {
    return (
      <Flex
        sx={{
          cursor: 'none',
          userSelect: 'none',
          position: 'relative',
          alignItems: 'center',
          justifyContent: 'center',
          flexDir: 'column',
          ...STYLES,
        }}
      >
        <Heading>{batchSelectionCount}</Heading>
        <Heading size="sm">Images</Heading>
      </Flex>
    );
  }

  if (props.dragData.payloadType === 'GALLERY_SELECTION') {
    return (
      <Flex
        sx={{
          cursor: 'none',
          userSelect: 'none',
          position: 'relative',
          alignItems: 'center',
          justifyContent: 'center',
          flexDir: 'column',
          ...STYLES,
        }}
      >
        <Heading>{gallerySelectionCount}</Heading>
        <Heading size="sm">Images</Heading>
      </Flex>
    );
  }

  return null;
};

export default memo(DragPreview);
