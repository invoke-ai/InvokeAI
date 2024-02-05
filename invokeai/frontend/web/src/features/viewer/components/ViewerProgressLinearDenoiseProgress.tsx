import { Image } from '@invoke-ai/ui-library';
import { useProgressImageRenderingStyles } from 'features/viewer/hooks/useProgressImageRenderingStyles';
import { memo } from 'react';
import type { ProgressImage } from 'services/events/types';

type Props = {
  progressImage: ProgressImage;
};

export const ViewerProgressLinearDenoiseProgress = memo(({ progressImage }: Props) => {
  const sx = useProgressImageRenderingStyles();

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
