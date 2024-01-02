import type { SystemStyleObject } from '@chakra-ui/react';
import { Image } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { memo, useMemo } from 'react';

const CurrentImagePreview = () => {
  const progress_image = useAppSelector(
    (state) => state.system.denoiseProgress?.progress_image
  );
  const shouldAntialiasProgressImage = useAppSelector(
    (state) => state.system.shouldAntialiasProgressImage
  );

  const sx = useMemo<SystemStyleObject>(
    () => ({
      imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
    }),
    [shouldAntialiasProgressImage]
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
      objectFit="contain"
      maxWidth="full"
      maxHeight="full"
      position="absolute"
      borderRadius="base"
      sx={sx}
    />
  );
};

export default memo(CurrentImagePreview);
