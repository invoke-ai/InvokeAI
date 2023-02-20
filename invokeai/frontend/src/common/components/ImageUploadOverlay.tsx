import { Heading } from '@chakra-ui/react';
import { useHotkeys } from 'react-hotkeys-hook';

type ImageUploadOverlayProps = {
  isDragAccept: boolean;
  isDragReject: boolean;
  overlaySecondaryText: string;
  setIsHandlingUpload: (isHandlingUpload: boolean) => void;
};

const ImageUploadOverlay = (props: ImageUploadOverlayProps) => {
  const {
    isDragAccept,
    isDragReject,
    overlaySecondaryText,
    setIsHandlingUpload,
  } = props;

  useHotkeys('esc', () => {
    setIsHandlingUpload(false);
  });

  return (
    <div className="dropzone-container">
      {isDragAccept && (
        <div className="dropzone-overlay is-drag-accept">
          <Heading size="lg">Upload Image{overlaySecondaryText}</Heading>
        </div>
      )}
      {isDragReject && (
        <div className="dropzone-overlay is-drag-reject">
          <Heading size="lg">Invalid Upload</Heading>
          <Heading size="md">Must be single JPEG or PNG image</Heading>
        </div>
      )}
    </div>
  );
};
export default ImageUploadOverlay;
