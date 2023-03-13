import { ImageUploaderTriggerContext } from 'app/contexts/ImageUploaderTriggerContext';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';
import IAIIconButton from './IAIIconButton';

const ImageUploaderIconButton = () => {
  const { t } = useTranslation();
  const openImageUploader = useContext(ImageUploaderTriggerContext);

  return (
    <IAIIconButton
      aria-label={t('accessibility.uploadImage')}
      tooltip="Upload Image"
      icon={<FaUpload />}
      onClick={openImageUploader || undefined}
    />
  );
};

export default ImageUploaderIconButton;
