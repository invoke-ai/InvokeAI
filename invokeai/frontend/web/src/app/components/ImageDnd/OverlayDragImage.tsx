import { Box, Image } from '@chakra-ui/react';
import { memo } from 'react';
import { ImageDTO } from 'services/api/types';

type OverlayDragImageProps = {
  image: ImageDTO;
};

const OverlayDragImage = (props: OverlayDragImageProps) => {
  return (
    <Box
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        userSelect: 'none',
        cursor: 'grabbing',
        opacity: 0.5,
      }}
    >
      <Image
        sx={{
          maxW: 36,
          maxH: 36,
          borderRadius: 'base',
          shadow: 'dark-lg',
        }}
        src={props.image.thumbnail_url}
      />
    </Box>
  );
};

export default memo(OverlayDragImage);
