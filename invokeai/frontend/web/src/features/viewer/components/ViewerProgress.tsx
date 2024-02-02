import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Image } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { latestLinearImageLoaded } from 'features/progress/store/progressSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiHourglassBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

export const ViewerProgress = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const linearDenoiseProgress = useAppSelector((s) => s.progress.linearDenoiseProgress);
  const linearLatestImageData = useAppSelector((s) => s.progress.linearLatestImageData);
  const shouldAntialiasProgressImage = useAppSelector((s) => s.system.shouldAntialiasProgressImage);

  const sx = useMemo<SystemStyleObject>(
    () => ({
      imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
    }),
    [shouldAntialiasProgressImage]
  );

  const shouldShowOutputImage = useMemo(() => {
    if (
      linearDenoiseProgress &&
      linearLatestImageData &&
      linearDenoiseProgress.graph_execution_state_id === linearLatestImageData.graph_execution_state_id
    ) {
      return true;
    }

    if (!linearDenoiseProgress?.progress_image && linearLatestImageData) {
      return true;
    }

    return false;
  }, [linearDenoiseProgress, linearLatestImageData]);

  const { data: imageDTO } = useGetImageDTOQuery(linearLatestImageData?.image_name ?? skipToken);

  const onLoad = useCallback(() => {
    dispatch(latestLinearImageLoaded());
  }, [dispatch]);

  if (shouldShowOutputImage && imageDTO) {
    return (
      <Image
        src={imageDTO.image_url}
        width={imageDTO.width}
        height={imageDTO.height}
        fallbackSrc={linearDenoiseProgress?.progress_image?.dataURL}
        draggable={false}
        data-testid="output-image"
        objectFit="contain"
        maxWidth="full"
        maxHeight="full"
        position="absolute"
        borderRadius="base"
        onLoad={onLoad}
      />
    );
  }

  if (linearDenoiseProgress?.progress_image) {
    return (
      <Image
        src={linearDenoiseProgress.progress_image.dataURL}
        width={linearDenoiseProgress.progress_image.width}
        height={linearDenoiseProgress.progress_image.height}
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
  }

  return <IAINoContentFallback icon={PiHourglassBold} label={t('viewer.noProgress')} />;
});

ViewerProgress.displayName = 'ViewerProgress';
