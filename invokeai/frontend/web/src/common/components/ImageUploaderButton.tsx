import { Heading } from '@chakra-ui/react';
import { ImageUploaderTriggerContext } from 'app/contexts/ImageUploaderTriggerContext';
import { useContext } from 'react';
import { FaUpload } from 'react-icons/fa';

type ImageUploaderButtonProps = {
  styleClass?: string;
};

const ImageUploaderButton = (props: ImageUploaderButtonProps) => {
  const { styleClass } = props;
  const open = useContext(ImageUploaderTriggerContext);

  const handleClickUpload = () => {
    open && open();
  };

  return (
    <div
      className={`image-uploader-button-outer ${styleClass}`}
      onClick={handleClickUpload}
    >
      <div className="image-upload-button">
        <FaUpload />
        <Heading size="lg">Click or Drag and Drop</Heading>
      </div>
    </div>
  );
};

export default ImageUploaderButton;
