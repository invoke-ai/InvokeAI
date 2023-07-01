import { Badge, Flex } from '@chakra-ui/react';
import { isString } from 'lodash-es';
import { useMemo } from 'react';
import { ImageDTO } from 'services/api/types';

type ImageMetadataOverlayProps = {
  image: ImageDTO;
};

const ImageMetadataOverlay = ({ image }: ImageMetadataOverlayProps) => {
  const model = useMemo(() => {
    if (!isString(image.metadata?.model)) {
      return;
    }

    return image.metadata?.model;
  }, [image.metadata]);

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
        {image.width} × {image.height}
      </Badge>
      {model && (
        <Badge variant="solid" colorScheme="base">
          {model}
        </Badge>
      )}
    </Flex>
  );
};

export default ImageMetadataOverlay;
