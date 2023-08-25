import { Badge, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import { ImageDTO } from 'services/api/types';

type ImageMetadataOverlayProps = {
  imageDTO: ImageDTO;
};

const ImageMetadataOverlay = ({ imageDTO }: ImageMetadataOverlayProps) => {
  return (
    <Flex
      sx={{
        pointerEvents: 'none',
        flexDirection: 'column',
        position: 'absolute',
        top: 0,
        insetInlineStart: 0,
        p: 2,
        alignItems: 'flex-start',
        gap: 2,
      }}
    >
      <Badge variant="solid" colorScheme="base">
        {imageDTO.width} × {imageDTO.height}
      </Badge>
    </Flex>
  );
};

export default memo(ImageMetadataOverlay);
