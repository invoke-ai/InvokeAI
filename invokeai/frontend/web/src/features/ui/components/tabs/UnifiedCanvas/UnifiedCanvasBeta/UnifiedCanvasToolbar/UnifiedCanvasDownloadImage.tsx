import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { canvasDownloadedAsImage } from 'features/canvas/store/actions';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaDownload } from 'react-icons/fa';

export default function UnifiedCanvasDownloadImage() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const canvasBaseLayer = getCanvasBaseLayer();

  const isStaging = useAppSelector(isStagingSelector);

  useHotkeys(
    ['shift+d'],
    () => {
      handleDownloadAsImage();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer]
  );

  const handleDownloadAsImage = () => {
    dispatch(canvasDownloadedAsImage());
  };

  return (
    <IAIIconButton
      aria-label={`${t('unifiedCanvas.downloadAsImage')} (Shift+D)`}
      tooltip={`${t('unifiedCanvas.downloadAsImage')} (Shift+D)`}
      icon={<FaDownload />}
      onClick={handleDownloadAsImage}
      isDisabled={isStaging}
    />
  );
}
