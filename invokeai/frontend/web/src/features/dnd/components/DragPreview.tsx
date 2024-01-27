import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, Flex, Heading, Image, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { TypesafeDraggableData } from 'features/dnd/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type OverlayDragImageProps = {
  dragData: TypesafeDraggableData | null;
};

const BOX_SIZE = 28;

const imageStyles: ChakraProps['sx'] = {
  w: BOX_SIZE,
  h: BOX_SIZE,
  maxW: BOX_SIZE,
  maxH: BOX_SIZE,
  shadow: 'dark-lg',
  borderRadius: 'lg',
  opacity: 0.3,
  borderColor: 'base.200',
  bg: 'base.900',
  color: 'base.100',
};

const multiImageStyles: ChakraProps['sx'] = {
  position: 'relative',
  alignItems: 'center',
  justifyContent: 'center',
  flexDir: 'column',
  ...imageStyles,
};

const DragPreview = (props: OverlayDragImageProps) => {
  const { t } = useTranslation();
  const selectionCount = useAppSelector((s) => s.gallery.selection.length);
  if (!props.dragData) {
    return null;
  }

  if (props.dragData.payloadType === 'NODE_FIELD') {
    const { field, fieldTemplate } = props.dragData.payload;
    return (
      <Box
        position="relative"
        p={2}
        px={3}
        opacity={0.7}
        bg="base.300"
        borderRadius="base"
        boxShadow="dark-lg"
        whiteSpace="nowrap"
        fontSize="sm"
      >
        <Text>{field.label || fieldTemplate.title}</Text>
      </Box>
    );
  }

  if (props.dragData.payloadType === 'IMAGE_DTO') {
    const { thumbnail_url, width, height } = props.dragData.payload.imageDTO;
    return (
      <Box position="relative" width="full" height="full" display="flex" alignItems="center" justifyContent="center">
        <Image sx={imageStyles} objectFit="contain" src={thumbnail_url} width={width} height={height} />
      </Box>
    );
  }

  if (props.dragData.payloadType === 'GALLERY_SELECTION') {
    return (
      <Flex sx={multiImageStyles}>
        <Heading>{selectionCount}</Heading>
        <Heading size="sm">{t('parameters.images')}</Heading>
      </Flex>
    );
  }

  return null;
};

export default memo(DragPreview);
