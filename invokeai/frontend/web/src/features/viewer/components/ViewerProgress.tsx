import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex , Image } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiHourglassBold } from 'react-icons/pi';

export const ViewerProgress = memo(() => {
  const { t } = useTranslation();
  const progress_image = useAppSelector((s) => s.system.denoiseProgress?.progress_image);
  const shouldAntialiasProgressImage = useAppSelector((s) => s.system.shouldAntialiasProgressImage);

  const sx = useMemo<SystemStyleObject>(
    () => ({
      imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
    }),
    [shouldAntialiasProgressImage]
  );

  if (!progress_image) {
    return <IAINoContentFallback icon={PiHourglassBold} label={t('viewer.noProgress')} />;
  }

  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center">
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
    </Flex>
  );
});

ViewerProgress.displayName = 'ViewerProgress';
