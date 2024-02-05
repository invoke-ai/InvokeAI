import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Image } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo, useMemo } from 'react';
import type { ProgressImage } from 'services/events/types';

type Props = {
  progressImage: ProgressImage;
};

export const ViewerProgressLinearDenoiseProgress = memo(({ progressImage }: Props) => {
  const shouldAntialiasProgressImage = useAppSelector((s) => s.system.shouldAntialiasProgressImage);

  const sx = useMemo<SystemStyleObject>(
    () => ({
      imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
    }),
    [shouldAntialiasProgressImage]
  );

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
});

ViewerProgressLinearDenoiseProgress.displayName = 'ViewerProgressLinearDenoiseProgress';
