import { Heading } from '@chakra-ui/react';
import { useContext } from 'react';
import { FaUpload } from 'react-icons/fa';
import { ImageUploaderTriggerContext } from 'app/contexts/ImageUploaderTriggerContext';

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
        <Heading size={'lg'}>Click or Drag and Drop</Heading>
      </div>
    </div>
  );
};

export default ImageUploaderButton;
