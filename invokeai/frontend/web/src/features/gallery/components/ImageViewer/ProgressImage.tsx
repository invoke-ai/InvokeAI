import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Image } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { memo, useMemo } from 'react';
import { $progressImage } from 'services/events/setEventListeners';

const CurrentImagePreview = () => {
  const progressImage = useStore($progressImage);
  const shouldAntialiasProgressImage = useAppSelector((s) => s.system.shouldAntialiasProgressImage);

  const sx = useMemo<SystemStyleObject>(
    () => ({
      imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
    }),
    [shouldAntialiasProgressImage]
  );

  if (!progressImage) {
    return null;
  }

  return (
    <Image
      src={progressImage.dataURL}
      width={progressImage.width}
      height={progressImage.height}
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
