import type { ChakraProps } from '@chakra-ui/react';
import { Box, Flex, Heading, Image } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';
import type { TypesafeDraggableData } from 'features/dnd/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

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
  opacity: 0.3,
  borderColor: 'base.200',
  bg: 'base.900',
  color: 'base.100',
};

const DragPreview = (props: OverlayDragImageProps) => {
  const { t } = useTranslation();
  if (!props.dragData) {
    return null;
  }

  if (props.dragData.payloadType === 'NODE_FIELD') {
    const { field, fieldTemplate } = props.dragData.payload;
    return (
      <Box
        sx={{
          position: 'relative',
          p: 2,
          px: 3,
          opacity: 0.7,
          bg: 'base.300',
          borderRadius: 'base',
          boxShadow: 'dark-lg',
          whiteSpace: 'nowrap',
          fontSize: 'sm',
        }}
      >
        <InvText>{field.label || fieldTemplate.title}</InvText>
      </Box>
    );
  }

  if (props.dragData.payloadType === 'IMAGE_DTO') {
    const { thumbnail_url, width, height } = props.dragData.payload.imageDTO;
    return (
      <Box
        sx={{
          position: 'relative',
          width: 'full',
          height: 'full',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
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

  if (props.dragData.payloadType === 'IMAGE_DTOS') {
    return (
      <Flex
        sx={{
          position: 'relative',
          alignItems: 'center',
          justifyContent: 'center',
          flexDir: 'column',
          ...STYLES,
        }}
      >
        <Heading>{props.dragData.payload.imageDTOs.length}</Heading>
        <Heading size="sm">{t('parameters.images')}</Heading>
      </Flex>
    );
  }

  return null;
};

export default memo(DragPreview);
