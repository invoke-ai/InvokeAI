import { Heading, useToast } from '@chakra-ui/react';
import { useCallback } from 'react';
import { FileRejection } from 'react-dropzone';
import { FaUpload } from 'react-icons/fa';
import ImageUploader from '../../features/options/ImageUploader';

interface InvokeImageUploaderProps {
  handleFile: (file: File) => void;
  styleClass?: string;
}

export default function InvokeImageUploader(props: InvokeImageUploaderProps) {
  const { handleFile, styleClass } = props;

  const toast = useToast();

  // Callbacks to for handling file upload attempts
  const fileAcceptedCallback = useCallback(
    (file: File) => handleFile(file),
    [handleFile]
  );

  const fileRejectionCallback = useCallback(
    (rejection: FileRejection) => {
      const msg = rejection.errors.reduce(
        (acc: string, cur: { message: string }) => acc + '\n' + cur.message,
        ''
      );

      toast({
        title: 'Upload failed',
        description: msg,
        status: 'error',
        isClosable: true,
      });
    },
    [toast]
  );

  return (
    <div className="image-upload-zone">
      <ImageUploader
        fileAcceptedCallback={fileAcceptedCallback}
        fileRejectionCallback={fileRejectionCallback}
        styleClass={
          styleClass
            ? `${styleClass} image-upload-child-wrapper`
            : `image-upload-child-wrapper`
        }
      >
        <div className="image-upload-child">
          <FaUpload size={'7rem'} />
          <Heading size={'lg'}>Upload or Drop Image Here</Heading>
        </div>
      </ImageUploader>
    </div>
  );
}
