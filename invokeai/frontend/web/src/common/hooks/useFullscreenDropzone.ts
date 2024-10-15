import { Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useEffect, useState } from 'react';
import type { Accept, FileRejection } from 'react-dropzone';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { useUploadImageMutation } from 'services/api/endpoints/images';
import type { PostUploadAction } from 'services/api/types';
import {  batchIndexIncremented, uploadingBatchChanged } from '../../features/gallery/store/gallerySlice';

const accept: Accept = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg', '.png'],
};

export const useFullscreenDropzone = () => {
  const { t } = useTranslation();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const [isHandlingUpload, setIsHandlingUpload] = useState<boolean>(false);
  const [uploadImage] = useUploadImageMutation();
  const activeTabName = useAppSelector(selectActiveTab);
  const dispatch = useAppDispatch()

  const getPostUploadAction = useCallback(
    (singleImage: boolean): PostUploadAction | undefined => {
      if (singleImage && activeTabName === 'upscaling') {
        return { type: 'SET_UPSCALE_INITIAL_IMAGE' };
      } else if (singleImage) {
        return { type: 'TOAST' };
      }
    },
    [activeTabName]
  );

  const fileRejectionCallback = useCallback(
    (rejection: FileRejection) => {
      setIsHandlingUpload(true);

      toast({
        id: 'UPLOAD_FAILED',
        title: t('toast.uploadFailed'),
        description: rejection.errors.map((error) => error.message).join('\n'),
        status: 'error',
      });
    },
    [t]
  );

  const fileAcceptedCallback = useCallback(
    (file: File, postUploadAction: PostUploadAction | undefined) => {
      uploadImage({
        file,
        image_category: 'user',
        is_intermediate: false,
        postUploadAction,
        board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
      });
    },
    [autoAddBoardId, uploadImage]
  );

  const onDrop = useCallback(
    (acceptedFiles: Array<File>, fileRejections: Array<FileRejection>) => {
      if (fileRejections.length > 1) {
        toast({
          id: 'UPLOAD_FAILED',
          title: t('toast.uploadFailed'),
          description: t('toast.uploadFailedInvalidUploadDesc'),
          status: 'error',
        });
        return;
      }

      fileRejections.forEach((rejection: FileRejection) => {
        fileRejectionCallback(rejection);
      });

      const postUploadAction = getPostUploadAction(acceptedFiles.length === 1);

      if (acceptedFiles.length === 1) {
        acceptedFiles.forEach((file: File) => {
          fileAcceptedCallback(file, postUploadAction);
        });
      } else {
        const batchTotal = acceptedFiles.length
        let index = 0
        // dispatch(uploadingBatchChanged({uploadingBatch: true, batchTotal: acceptedFiles.length}))
     

      toast({
        id: 'BATCH_UPLOADING',
        title: "Batch uploading",
        status: 'info',
        updateDescription: true,
        description: `Uploading ${index} or ${batchTotal}`,
        duration: null
      });

      acceptedFiles.forEach((file: File) => {
        fileAcceptedCallback(file, undefined);
        toast({
          id: 'BATCH_UPLOADING',
          title: "Batch uploading",
          status: 'info',
          updateDescription: true,
          description: `Uploading ${index} of ${batchTotal}`,
          duration: null
        });
      });
    }
    },
    [t, fileAcceptedCallback, fileRejectionCallback, getPostUploadAction]
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


