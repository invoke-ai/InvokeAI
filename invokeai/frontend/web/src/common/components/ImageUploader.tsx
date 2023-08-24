import { Box } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppToaster } from 'app/components/Toaster';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { AnimatePresence, motion } from 'framer-motion';
import {
  KeyboardEvent,
  ReactNode,
  memo,
  useCallback,
  useEffect,
  useState,
} from 'react';
import { FileRejection, useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { useUploadImageMutation } from 'services/api/endpoints/images';
import { PostUploadAction } from 'services/api/types';
import ImageUploadOverlay from './ImageUploadOverlay';

const selector = createSelector(
  [stateSelector, activeTabNameSelector],
  ({ gallery }, activeTabName) => {
    let postUploadAction: PostUploadAction = { type: 'TOAST' };

    if (activeTabName === 'unifiedCanvas') {
      postUploadAction = { type: 'SET_CANVAS_INITIAL_IMAGE' };
    }

    if (activeTabName === 'img2img') {
      postUploadAction = { type: 'SET_INITIAL_IMAGE' };
    }

    const { autoAddBoardId } = gallery;

    return {
      autoAddBoardId,
      postUploadAction,
    };
  },
  defaultSelectorOptions
);

type ImageUploaderProps = {
  children: ReactNode;
};

const ImageUploader = (props: ImageUploaderProps) => {
  const { children } = props;
  const { autoAddBoardId, postUploadAction } = useAppSelector(selector);
  const isBusy = useAppSelector(selectIsBusy);
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const [isHandlingUpload, setIsHandlingUpload] = useState<boolean>(false);

  const [uploadImage] = useUploadImageMutation();

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
      uploadImage({
        file,
        image_category: 'user',
        is_intermediate: false,
        postUploadAction,
        board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
      });
    },
    [autoAddBoardId, postUploadAction, uploadImage]
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
  } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    noClick: true,
    onDrop,
    onDragOver: () => setIsHandlingUpload(true),
    disabled: isBusy,
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

    // Add the paste event listener
    document.addEventListener('paste', handlePaste);

    return () => {
      document.removeEventListener('paste', handlePaste);
    };
  }, [inputRef]);

  return (
    <Box
      {...getRootProps({ style: {} })}
      onKeyDown={(e: KeyboardEvent) => {
        // Bail out if user hits spacebar - do not open the uploader
        if (e.key === ' ') {
          return;
        }
      }}
    >
      <input {...getInputProps()} />
      {children}
      <AnimatePresence>
        {isDragActive && isHandlingUpload && (
          <motion.div
            key="image-upload-overlay"
            initial={{
              opacity: 0,
            }}
            animate={{
              opacity: 1,
              transition: { duration: 0.1 },
            }}
            exit={{
              opacity: 0,
              transition: { duration: 0.1 },
            }}
          >
            <ImageUploadOverlay
              isDragAccept={isDragAccept}
              isDragReject={isDragReject}
              setIsHandlingUpload={setIsHandlingUpload}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  );
};

export default memo(ImageUploader);
