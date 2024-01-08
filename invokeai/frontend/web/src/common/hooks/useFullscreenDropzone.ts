import { useAppToaster } from 'app/components/Toaster';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback, useEffect, useState } from 'react';
import type { Accept, FileRejection } from 'react-dropzone';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { useUploadImageMutation, useUploadMultipleImagesMutation } from 'services/api/endpoints/images';
import type { PostUploadAction } from 'services/api/types';

const accept: Accept = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg', '.png'],
};

const selectPostUploadAction = createMemoizedSelector(
  activeTabNameSelector,
  (activeTabName) => {
    let postUploadAction: PostUploadAction = { type: 'TOAST' };

    if (activeTabName === 'unifiedCanvas') {
      postUploadAction = { type: 'SET_CANVAS_INITIAL_IMAGE' };
    }

    if (activeTabName === 'img2img') {
      postUploadAction = { type: 'SET_INITIAL_IMAGE' };
    }

    return postUploadAction;
  }
);

export const useFullscreenDropzone = () => {
  const { t } = useTranslation();
  const toaster = useAppToaster();
  const postUploadAction = useAppSelector(selectPostUploadAction);
  const autoAddBoardId = useAppSelector((s) => s.gallery.autoAddBoardId);
  const [isHandlingUpload, setIsHandlingUpload] = useState<boolean>(false);

  const [uploadImage] = useUploadImageMutation();
  const [uploadMultipleImages] = useUploadMultipleImagesMutation()

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

  const filesAcceptedCallback = useCallback(
    async (files: Array<File>) => {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file); // Use 'files' as the key for each file
      });

      console.log('image object in hook to thunk: ');
      console.log(files)

      uploadMultipleImages({
            formData,
            image_category: 'user',
            is_intermediate: false,
            postUploadAction,
            board_id: autoAddBoardId === 'none' ? undefined: autoAddBoardId,
        });
    },
    [autoAddBoardId, postUploadAction, uploadMultipleImages]
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
      if (fileRejections.length > 99) {
        // number of files allowed to upload at once
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

      if (acceptedFiles.length > 1) {
        console.log("multiple files uploaded")
        filesAcceptedCallback(acceptedFiles);
      } else {
        console.log("single file uploaded")
        acceptedFiles.forEach((file: File) => {
            fileAcceptedCallback(file);
          });
      }
    },
    [t, toaster, filesAcceptedCallback, fileAcceptedCallback, fileRejectionCallback]
  );

  const onDragOver = useCallback(() => {
    setIsHandlingUpload(true);
  }, []);

  const dropzone = useDropzone({
    accept,
    noClick: true,
    onDrop,
    onDragOver,
    multiple: true,
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
        dropzone.inputRef.current?.dispatchEvent(
          new Event('change', { bubbles: true })
        );
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
