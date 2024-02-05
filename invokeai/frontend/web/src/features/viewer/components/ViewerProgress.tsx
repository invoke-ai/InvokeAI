import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { ViewerProgressLatestImage } from 'features/viewer/components/ViewerProgressLatestImage';
import { ViewerProgressLinearDenoiseProgress } from 'features/viewer/components/ViewerProgressLinearDenoiseProgress';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiHourglassBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

export const ViewerProgress = memo(() => {
  const { t } = useTranslation();
  const linearDenoiseProgress = useAppSelector((s) => s.progress.linearDenoiseProgress);
  const linearLatestImageData = useAppSelector((s) => s.progress.linearLatestImageData);

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

  if (shouldShowOutputImage && imageDTO) {
    return <ViewerProgressLatestImage imageDTO={imageDTO} />;
  }

  if (linearDenoiseProgress?.progress_image) {
    return <ViewerProgressLinearDenoiseProgress progressImage={linearDenoiseProgress.progress_image} />;
  }

  return <IAINoContentFallback icon={PiHourglassBold} label={t('viewer.noProgress')} />;
});

ViewerProgress.displayName = 'ViewerProgress';
