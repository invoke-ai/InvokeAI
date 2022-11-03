import { useCallback, ReactNode, useState, useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { FileRejection, useDropzone } from 'react-dropzone';
import { useToast } from '@chakra-ui/react';
import { uploadImage } from '../../app/socketio/actions';
import { ImageUploadDestination, UploadImagePayload } from '../../app/invokeai';
import { ImageUploaderTriggerContext } from '../../app/contexts/ImageUploaderTriggerContext';
import { activeTabNameSelector } from '../../features/options/optionsSelectors';
import { tabDict } from '../../features/tabs/InvokeTabs';
import ImageUploadOverlay from './ImageUploadOverlay';

type ImageUploaderProps = {
  children: ReactNode;
};

const ImageUploader = (props: ImageUploaderProps) => {
  const { children } = props;
  const dispatch = useAppDispatch();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const toast = useToast({});
  const [isHandlingUpload, setIsHandlingUpload] = useState<boolean>(false);

  const fileRejectionCallback = useCallback(
    (rejection: FileRejection) => {
      setIsHandlingUpload(true);
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

  const fileAcceptedCallback = useCallback(
    (file: File) => {
      setIsHandlingUpload(true);
      const payload: UploadImagePayload = { file };
      if (['img2img', 'inpainting'].includes(activeTabName)) {
        payload.destination = activeTabName as ImageUploadDestination;
      }
      dispatch(uploadImage(payload));
    },
    [dispatch, activeTabName]
  );

  const onDrop = useCallback(
    (acceptedFiles: Array<File>, fileRejections: Array<FileRejection>) => {
      fileRejections.forEach((rejection: FileRejection) => {
        fileRejectionCallback(rejection);
      });

      acceptedFiles.forEach((file: File) => {
        fileAcceptedCallback(file);
      });
    },
    [fileAcceptedCallback, fileRejectionCallback]
  );

  const {
    getRootProps,
    getInputProps,
    isDragAccept,
    isDragReject,
    isDragActive,
    open,
  } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    noClick: true,
    onDrop,
    onDragOver: () => setIsHandlingUpload(true),
    maxFiles: 1,
  });

  useEffect(() => {
    const pasteImageListener = (e: ClipboardEvent) => {
      const dataTransferItemList = e.clipboardData?.items;
      if (!dataTransferItemList) return;

      const imageItems: Array<DataTransferItem> = [];

      for (const item of dataTransferItemList) {
        if (
          item.kind === 'file' &&
          ['image/png', 'image/jpg'].includes(item.type)
        ) {
          imageItems.push(item);
        }
      }

      if (!imageItems.length) return;

      e.stopImmediatePropagation();

      if (imageItems.length > 1) {
        toast({
          description:
            'Multiple images pasted, may only upload one image at a time',
          status: 'error',
          isClosable: true,
        });
        return;
      }

      const file = imageItems[0].getAsFile();

      if (!file) {
        toast({
          description: 'Unable to load file',
          status: 'error',
          isClosable: true,
        });
        return;
      }

      const payload: UploadImagePayload = { file };
      if (['img2img', 'inpainting'].includes(activeTabName)) {
        payload.destination = activeTabName as ImageUploadDestination;
      }

      dispatch(uploadImage(payload));
    };
    document.addEventListener('paste', pasteImageListener);
    return () => {
      document.removeEventListener('paste', pasteImageListener);
    };
  }, [dispatch, toast, activeTabName]);

  const overlaySecondaryText = ['img2img', 'inpainting'].includes(activeTabName)
    ? ` to ${tabDict[activeTabName as keyof typeof tabDict].tooltip}`
    : ``;

  return (
    <ImageUploaderTriggerContext.Provider value={open}>
      <div {...getRootProps({ style: {} })}>
        <input {...getInputProps()} />
        {children}
        {isDragActive && isHandlingUpload && (
          <ImageUploadOverlay
            isDragAccept={isDragAccept}
            isDragReject={isDragReject}
            overlaySecondaryText={overlaySecondaryText}
            setIsHandlingUpload={setIsHandlingUpload}
          />
        )}
      </div>
    </ImageUploaderTriggerContext.Provider>
  );
};

export default ImageUploader;
