import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';
import IAIIconButton from './IAIIconButton';
import useImageUploader from 'common/hooks/useImageUploader';

const ImageUploaderIconButton = () => {
  const { t } = useTranslation();
  const { openUploader } = useImageUploader();

  return (
    <IAIIconButton
      aria-label={t('accessibility.uploadImage')}
      tooltip="Upload Image"
      icon={<FaUpload />}
      onClick={openUploader}
    />
  );
};

export default ImageUploaderIconButton;
