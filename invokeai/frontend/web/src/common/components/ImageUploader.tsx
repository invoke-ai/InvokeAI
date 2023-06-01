import { Box } from '@chakra-ui/react';
import { ImageUploaderTriggerContext } from 'app/contexts/ImageUploaderTriggerContext';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import useImageUploader from 'common/hooks/useImageUploader';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { ResourceKey } from 'i18next';
import {
  KeyboardEvent,
  memo,
  ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { FileRejection, useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { imageUploaded } from 'services/thunks/image';
import ImageUploadOverlay from './ImageUploadOverlay';
import { useAppToaster } from 'app/components/Toaster';
import { filter, map, some } from 'lodash-es';
import { createSelector } from '@reduxjs/toolkit';
import { systemSelector } from 'features/system/store/systemSelectors';
import { ErrorCode } from 'react-dropzone';

const selector = createSelector(
  [systemSelector, activeTabNameSelector],
  (system, activeTabName) => {
    const { isConnected, isUploading } = system;

    const isUploaderDisabled = !isConnected || isUploading;

    return {
      isUploaderDisabled,
      activeTabName,
    };
  }
);

type ImageUploaderProps = {
  children: ReactNode;
};

const ImageUploader = (props: ImageUploaderProps) => {
  const { children } = props;
  const dispatch = useAppDispatch();
  const { isUploaderDisabled, activeTabName } = useAppSelector(selector);
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const [isHandlingUpload, setIsHandlingUpload] = useState<boolean>(false);
  const { setOpenUploaderFunction } = useImageUploader();

  const fileRejectionCallback = useCallback(
    (rejection: FileRejection) => {
      setIsHandlingUpload(true);

      toaster({
        title: t('toast.uploadFailed'),
        description: rejection.errors.map((error) => error.message).join('\n'),
        status: 'error',
      });
    },
    [t, toaster]
  );

  const fileAcceptedCallback = useCallback(
    async (file: File) => {
      dispatch(
        imageUploaded({
          formData: { file },
          imageCategory: 'user',
          isIntermediate: false,
        })
      );
    },
    [dispatch]
  );

  const onDrop = useCallback(
    (acceptedFiles: Array<File>, fileRejections: Array<FileRejection>) => {
      if (fileRejections.length > 1) {
        toaster({
          title: t('toast.uploadFailed'),
          description: t('toast.uploadFailedInvalidUploadDesc'),
          status: 'error',
        });
        return;
      }

      fileRejections.forEach((rejection: FileRejection) => {
        fileRejectionCallback(rejection);
      });

      acceptedFiles.forEach((file: File) => {
        fileAcceptedCallback(file);
      });
    },
    [t, toaster, fileAcceptedCallback, fileRejectionCallback]
  );

  const {
    getRootProps,
    getInputProps,
    isDragAccept,
    isDragReject,
    isDragActive,
    inputRef,
    open,
  } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    noClick: true,
    onDrop,
    onDragOver: () => setIsHandlingUpload(true),
    disabled: isUploaderDisabled,
    multiple: false,
  });

  useEffect(() => {
    // This is a hack to allow pasting images into the uploader
    const handlePaste = async (e: ClipboardEvent) => {
      if (!inputRef.current) {
        return;
      }

      if (e.clipboardData?.files) {
        // Set the files on the inputRef
        inputRef.current.files = e.clipboardData.files;
        // Dispatch the change event, dropzone catches this and we get to use its own validation
        inputRef.current?.dispatchEvent(new Event('change', { bubbles: true }));
      }
    };

    // Set the open function so we can open the uploader from anywhere
    setOpenUploaderFunction(open);

    // Add the paste event listener
    document.addEventListener('paste', handlePaste);

    return () => {
      document.removeEventListener('paste', handlePaste);
      setOpenUploaderFunction(() => {
        return;
      });
    };
  }, [inputRef, open, setOpenUploaderFunction]);

  return (
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
          setIsHandlingUpload={setIsHandlingUpload}
        />
      )}
    </Box>
  );
};

export default memo(ImageUploader);
