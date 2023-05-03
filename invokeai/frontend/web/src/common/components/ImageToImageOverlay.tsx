import { Badge, Box, Flex } from '@chakra-ui/react';
import { Image } from 'app/types/invokeai';

type ImageToImageOverlayProps = {
  image: Image;
};

const ImageToImageOverlay = ({ image }: ImageToImageOverlayProps) => {
  return (
    <Box
      sx={{
        top: 0,
        left: 0,
        w: 'full',
        h: 'full',
        position: 'absolute',
      }}
    >
      <Flex
        sx={{
          position: 'absolute',
          top: 0,
          right: 0,
          p: 2,
          alignItems: 'flex-start',
        }}
      >
        <Badge variant="solid" colorScheme="base">
          {image.metadata?.width} × {image.metadata?.height}
        </Badge>
      </Flex>
    </Box>
  );
};

export default ImageToImageOverlay;
