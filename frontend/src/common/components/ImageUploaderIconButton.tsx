import { useContext } from 'react';
import { FaUpload } from 'react-icons/fa';
import { ImageUploaderTriggerContext } from '../../app/contexts/ImageUploaderTriggerContext';
import IAIIconButton from './IAIIconButton';

const ImageUploaderIconButton = () => {
  const openImageUploader = useContext(ImageUploaderTriggerContext);

  return (
    <IAIIconButton
      aria-label="Upload Image"
      tooltip="Upload Image"
      icon={<FaUpload />}
      onClick={openImageUploader || undefined}
    />
  );
};

export default ImageUploaderIconButton;
