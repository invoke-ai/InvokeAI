import { useAppToaster } from 'app/components/Toaster';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback, useEffect, useState } from 'react';
import type { Accept, FileRejection } from 'react-dropzone';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { useUploadImageMutation } from 'services/api/endpoints/images';
import type { PostUploadAction } from 'services/api/types';

const accept: Accept = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg', '.png'],
};

const selectPostUploadAction = createMemoizedSelector(activeTabNameSelector, (activeTabName) => {
  let postUploadAction: PostUploadAction = { type: 'TOAST' };

  if (activeTabName === 'unifiedCanvas') {
    postUploadAction = { type: 'SET_CANVAS_INITIAL_IMAGE' };
  }

  if (activeTabName === 'img2img') {
    postUploadAction = { type: 'SET_INITIAL_IMAGE' };
  }

  return postUploadAction;
});

export const useFullscreenDropzone = () => {
  const { t } = useTranslation();
  const toaster = useAppToaster();
  const postUploadAction = useAppSelector(selectPostUploadAction);
  const autoAddBoardId = useAppSelector((s) => s.gallery.autoAddBoardId);
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

  const onDragOver = useCallback(() => {
    setIsHandlingUpload(true);
  }, []);

  const dropzone = useDropzone({
    accept,
    noClick: true,
    onDrop,
    onDragOver,
    multiple: false,
    noKeyboard: true,
  });

  useEffect(() => {
    // This is a hack to allow pasting images into the uploader
    const handlePaste = async (e: ClipboardEvent) => {
      if (!dropzone.inputRef.current) {
        return;
      }

      if (e.clipboardData?.files) {
        // Set the files on the dropzone.inputRef
        dropzone.inputRef.current.files = e.clipboardData.files;
        // Dispatch the change event, dropzone catches this and we get to use its own validation
        dropzone.inputRef.current?.dispatchEvent(new Event('change', { bubbles: true }));
      }
    };

    // Add the paste event listener
    document.addEventListener('paste', handlePaste);

    return () => {
      document.removeEventListener('paste', handlePaste);
    };
  }, [dropzone.inputRef]);

  return { dropzone, isHandlingUpload, setIsHandlingUpload };
};
