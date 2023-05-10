import { useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import useImageUploader from 'common/hooks/useImageUploader';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';

export default function UnifiedCanvasFileUploader() {
  const isStaging = useAppSelector(isStagingSelector);
  const { openUploader } = useImageUploader();
  const { t } = useTranslation();

  return (
    <IAIIconButton
      aria-label={t('common.upload')}
      tooltip={t('common.upload')}
      icon={<FaUpload />}
      onClick={openUploader}
      isDisabled={isStaging}
    />
  );
}
