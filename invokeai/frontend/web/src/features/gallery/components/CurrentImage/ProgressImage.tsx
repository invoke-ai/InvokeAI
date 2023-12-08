import { Image } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';

const CurrentImagePreview = () => {
  const progress_image = useAppSelector(
    (state) => state.system.denoiseProgress?.progress_image
  );
  const shouldAntialiasProgressImage = useAppSelector(
    (state) => state.system.shouldAntialiasProgressImage
  );

  if (!progress_image) {
    return null;
  }

  return (
    <Image
      src={progress_image.dataURL}
      width={progress_image.width}
      height={progress_image.height}
      draggable={false}
      data-testid="progress-image"
      sx={{
        objectFit: 'contain',
        maxWidth: 'full',
        maxHeight: 'full',
        height: 'auto',
        position: 'absolute',
        borderRadius: 'base',
        imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
      }}
    />
  );
};

export default memo(CurrentImagePreview);
