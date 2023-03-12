import { Box, useToast } from '@chakra-ui/react';
import { ImageUploaderTriggerContext } from 'app/contexts/ImageUploaderTriggerContext';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import useImageUploader from 'common/hooks/useImageUploader';
import { uploadImage } from 'features/gallery/store/thunks/uploadImage';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { ResourceKey } from 'i18next';
import {
  KeyboardEvent,
  memo,
  ReactNode,
  useCallback,
  useEffect,
  useState,
} from 'react';
import { FileRejection, useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import ImageUploadOverlay from './ImageUploadOverlay';

type ImageUploaderProps = {
  children: ReactNode;
};

const ImageUploader = (props: ImageUploaderProps) => {
  const { children } = props;
  const dispatch = useAppDispatch();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const toast = useToast({});
  const { t } = useTranslation();
  const [isHandlingUpload, setIsHandlingUpload] = useState<boolean>(false);
  const { setOpenUploader } = useImageUploader();

  const fileRejectionCallback = useCallback(
    (rejection: FileRejection) => {
      setIsHandlingUpload(true);
      const msg = rejection.errors.reduce(
        (acc: string, cur: { message: string }) => `${acc}\n${cur.message}`,
        ''
      );
      toast({
        title: t('toast.uploadFailed'),
        description: msg,
        status: 'error',
        isClosable: true,
      });
    },
    [t, toast]
  );

  const fileAcceptedCallback = useCallback(
    async (file: File) => {
      dispatch(uploadImage({ imageFile: file }));
    },
    [dispatch]
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

  setOpenUploader(open);

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
          description: t('toast.uploadFailedMultipleImagesDesc'),
          status: 'error',
          isClosable: true,
        });
        return;
      }

      const file = imageItems[0].getAsFile();

      if (!file) {
        toast({
          description: t('toast.uploadFailedUnableToLoadDesc'),
          status: 'error',
          isClosable: true,
        });
        return;
      }

      dispatch(uploadImage({ imageFile: file }));
    };
    document.addEventListener('paste', pasteImageListener);
    return () => {
      document.removeEventListener('paste', pasteImageListener);
    };
  }, [t, dispatch, toast, activeTabName]);

  const overlaySecondaryText = ['img2img', 'unifiedCanvas'].includes(
    activeTabName
  )
    ? ` to ${String(t(`common.${activeTabName}` as ResourceKey))}`
    : ``;

  return (
    <ImageUploaderTriggerContext.Provider value={open}>
      <Box
        {...getRootProps({ style: {} })}
        onKeyDown={(e: KeyboardEvent) => {
          // Bail out if user hits spacebar - do not open the uploader
          if (e.key === ' ') return;
        }}
      >
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
      </Box>
    </ImageUploaderTriggerContext.Provider>
  );
};

export default memo(ImageUploader);
