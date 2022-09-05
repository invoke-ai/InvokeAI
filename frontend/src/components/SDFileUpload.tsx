import { useToast } from '@chakra-ui/react';
import { base64ArrayBuffer } from '../util/base64ArrayBuffer';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useAppDispatch } from '../app/hooks';
import { addImage } from '../app/sdSlice';
import SDButton from './SDButton';

const SDFileUpload = () => {
  const toast = useToast();
  const dispatch = useAppDispatch();

  const onDrop = useCallback(
    (acceptedFiles: Array<File>, fileRejections: any) => {
      fileRejections.forEach((rejection: any) => {
        const msg = rejection.errors.reduce(
          (acc: string, cur: { message: string }) => acc + '\n' + cur.message,
          ''
        );

        toast({
          title: 'Upload failed.',
          description: msg,
          status: 'error',
          isClosable: true,
        });
      });
      acceptedFiles.forEach((file: File) => {
        const reader = new FileReader();

        reader.onabort = () =>
          toast({
            title: 'Upload aborted.',
            status: 'error',
            isClosable: true,
          });

        reader.onerror = () =>
          toast({
            title: `Upload failed.`,
            status: 'error',
            isClosable: true,
          });

        reader.onload = () => {
          const binaryStr = reader.result;
          const base64 = base64ArrayBuffer(binaryStr);
          const uri = 'data:image/image/png;base64,' + base64;
          dispatch(addImage(uri));
        };
        reader.readAsArrayBuffer(file);
      });
    },
    []
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg', '.png'],
    },
  });

  return (
    <div {...getRootProps()}>
      <input {...getInputProps()} />
      <SDButton
        label='Drag and Drop / Click to Upload'
        colorScheme={isDragActive ? 'orange' : 'yellow'}
      />
    </div>
  );
};

export default SDFileUpload;
