import { useAppSelector } from 'app/store/storeHooks';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { selectMaxImageUploadCount } from 'features/system/store/configSlice';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useUploadImageMutation } from 'services/api/endpoints/images';
import type { PostUploadAction } from 'services/api/types';

type UseImageUploadButtonArgs = {
  postUploadAction?: PostUploadAction;
  isDisabled?: boolean;
  allowMultiple?: boolean;
};

/**
 * Provides image uploader functionality to any component.
 *
 * @example
 * const { getUploadButtonProps, getUploadInputProps, openUploader } = useImageUploadButton({
 *   postUploadAction: {
 *     type: 'SET_CONTROL_ADAPTER_IMAGE',
 *     controlNetId: '12345',
 *   },
 *   isDisabled: getIsUploadDisabled(),
 * });
 *
 * // open the uploaded directly
 * const handleSomething = () => { openUploader() }
 *
 * // in the render function
 * <Button {...getUploadButtonProps()} /> // will open the file dialog on click
 * <input {...getUploadInputProps()} /> // hidden, handles native upload functionality
 */
export const useImageUploadButton = ({
  postUploadAction,
  isDisabled,
  allowMultiple = false,
}: UseImageUploadButtonArgs) => {
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const [uploadImage] = useUploadImageMutation();
  const maxImageUploadCount = useAppSelector(selectMaxImageUploadCount);

  const onDropAccepted = useCallback(
    (files: File[]) => {
      for (const file of files) {
        uploadImage({
          file,
          image_category: 'user',
          is_intermediate: false,
          postUploadAction: postUploadAction ?? { type: 'TOAST' },
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
        });
      }
    },
    [autoAddBoardId, postUploadAction, uploadImage]
  );

  const {
    getRootProps: getUploadButtonProps,
    getInputProps: getUploadInputProps,
    open: openUploader,
  } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    onDropAccepted,
    disabled: isDisabled,
    noDrag: true,
    multiple: allowMultiple && (maxImageUploadCount === undefined || maxImageUploadCount > 1),
    maxFiles: maxImageUploadCount,
  });

  return { getUploadButtonProps, getUploadInputProps, openUploader };
};
