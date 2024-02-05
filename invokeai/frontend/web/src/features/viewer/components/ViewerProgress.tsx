import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { ViewerProgressLatestImage } from 'features/viewer/components/ViewerProgressLatestImage';
import { ViewerProgressLinearDenoiseProgress } from 'features/viewer/components/ViewerProgressLinearDenoiseProgress';
import { useLatestImageDTO } from 'features/viewer/hooks/useLatestImageDTO';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiHourglassBold } from 'react-icons/pi';

export const ViewerProgress = memo(() => {
  const { t } = useTranslation();
  const linearDenoiseProgress = useAppSelector((s) => s.progress.linearDenoiseProgress);
  const latestImageDTO = useLatestImageDTO();

  if (latestImageDTO) {
    return <ViewerProgressLatestImage imageDTO={latestImageDTO} />;
  }

  if (linearDenoiseProgress?.progress_image) {
    return <ViewerProgressLinearDenoiseProgress progressImage={linearDenoiseProgress.progress_image} />;
  }

  return <IAINoContentFallback icon={PiHourglassBold} label={t('viewer.noProgress')} />;
});

ViewerProgress.displayName = 'ViewerProgress';
