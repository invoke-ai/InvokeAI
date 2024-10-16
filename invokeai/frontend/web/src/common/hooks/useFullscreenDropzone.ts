import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useEffect, useState } from 'react';
import type { Accept, FileRejection } from 'react-dropzone';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { useUploadImageMutation } from 'services/api/endpoints/images';
import type { PostUploadAction } from 'services/api/types';

const log = logger('gallery');

const accept: Accept = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg', '.png'],
};

export const useFullscreenDropzone = () => {
  useAssertSingleton('useFullscreenDropzone');
  const { t } = useTranslation();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const [isHandlingUpload, setIsHandlingUpload] = useState<boolean>(false);
  const [uploadImage] = useUploadImageMutation();
  const activeTabName = useAppSelector(selectActiveTab);

  const getPostUploadAction = useCallback(
    (isSingleImage: boolean, isLastImage: boolean): PostUploadAction => {
      if (isSingleImage && activeTabName === 'upscaling') {
        return { type: 'SET_UPSCALE_INITIAL_IMAGE' };
      } else if (isSingleImage || isLastImage) {
        // Omit the duration if it's the last image - this allows the toast to auto-close
        return { type: 'TOAST' };
      } else {
        // Set duration to `null` to prevent auto-close on any toast that is not the last image
        return { type: 'TOAST', duration: null };
      }
    },
    [activeTabName]
  );

  const onDrop = useCallback(
    (acceptedFiles: Array<File>, fileRejections: Array<FileRejection>) => {
      if (fileRejections.length > 0) {
        const errors = fileRejections.map((rejection) => ({
          errors: rejection.errors.map(({ message }) => message),
          file: rejection.file.path,
        }));
        log.error({ errors }, 'Invalid upload');
        toast({
          id: 'UPLOAD_FAILED',
          title: t('toast.uploadFailed'),
          description: t('toast.uploadFailedInvalidUploadDesc'),
          status: 'error',
        });
        return;
      }

      const isSingleImage = acceptedFiles.length === 1;

      for (const [i, file] of acceptedFiles.entries()) {
        const isLastImage = i === acceptedFiles.length - 1;
        uploadImage({
          file,
          image_category: 'user',
          is_intermediate: false,
          postUploadAction: getPostUploadAction(isSingleImage, isLastImage),
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
          // The `imageUploaded` listener does some extra logic, like switching to the asset view on upload on the
          // first upload of a "batch".
          isFirstUploadOfBatch: i === 0,
        });
      }
    },
    [t, uploadImage, getPostUploadAction, autoAddBoardId]
  );

  const onDragOver = useCallback(() => {
    setIsHandlingUpload(true);
  }, []);

  const dropzone = useDropzone({
    accept,
    noClick: true,
    onDrop,
    onDragOver,
    noKeyboard: true,
  });

  useEffect(() => {
    // This is a hack to allow pasting images into the uploader
    const handlePaste = (e: ClipboardEvent) => {
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
