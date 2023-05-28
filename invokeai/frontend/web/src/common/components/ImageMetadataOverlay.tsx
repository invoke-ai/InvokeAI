import { Badge, Flex } from '@chakra-ui/react';
import { isNumber, isString } from 'lodash-es';
import { useMemo } from 'react';
import { ImageDTO } from 'services/api';

type ImageMetadataOverlayProps = {
  image: ImageDTO;
};

const ImageMetadataOverlay = ({ image }: ImageMetadataOverlayProps) => {
  const dimensions = useMemo(() => {
    if (!isNumber(image.metadata?.width) || isNumber(!image.metadata?.height)) {
      return;
    }

    return `${image.metadata?.width} × ${image.metadata?.height}`;
  }, [image.metadata]);

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
        right: 0,
        p: 2,
        alignItems: 'flex-end',
        gap: 2,
      }}
    >
      {dimensions && (
        <Badge variant="solid" colorScheme="base">
          {dimensions}
        </Badge>
      )}
      {model && (
        <Badge variant="solid" colorScheme="base">
          {model}
        </Badge>
      )}
    </Flex>
  );
};

export default ImageMetadataOverlay;
