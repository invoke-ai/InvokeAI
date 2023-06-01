import { Image } from '@chakra-ui/react';
import { memo } from 'react';
import { ImageDTO } from 'services/api';

type OverlayDragImageProps = {
  image: ImageDTO;
};

const OverlayDragImage = (props: OverlayDragImageProps) => {
  return (
    <Image
      sx={{
        maxW: 36,
        maxH: 36,
        borderRadius: 'base',
        shadow: 'dark-lg',
      }}
      src={props.image.thumbnail_url}
    />
  );
};

export default memo(OverlayDragImage);
