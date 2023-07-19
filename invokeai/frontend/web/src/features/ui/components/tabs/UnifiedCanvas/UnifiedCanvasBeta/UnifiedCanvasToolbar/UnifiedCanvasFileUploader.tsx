import { useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';

export default function UnifiedCanvasFileUploader() {
  const isStaging = useAppSelector(isStagingSelector);

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    postUploadAction: { type: 'SET_CANVAS_INITIAL_IMAGE' },
  });
  const { t } = useTranslation();

  return (
    <>
      <IAIIconButton
        aria-label={t('common.upload')}
        tooltip={t('common.upload')}
        icon={<FaUpload />}
        isDisabled={isStaging}
        {...getUploadButtonProps()}
      />
      <input {...getUploadInputProps()} />
    </>
  );
}
