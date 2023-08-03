import { useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useUploadImageMutation } from 'services/api/endpoints/images';
import { PostUploadAction } from 'services/api/types';

type UseImageUploadButtonArgs = {
  postUploadAction?: PostUploadAction;
  isDisabled?: boolean;
};

/**
 * Provides image uploader functionality to any component.
 *
 * @example
 * const { getUploadButtonProps, getUploadInputProps, openUploader } = useImageUploadButton({
 *   postUploadAction: {
 *     type: 'SET_CONTROLNET_IMAGE',
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
}: UseImageUploadButtonArgs) => {
  const autoAddBoardId = useAppSelector(
    (state) => state.gallery.autoAddBoardId
  );
  const [uploadImage] = useUploadImageMutation();
  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];

      if (!file) {
        return;
      }

      uploadImage({
        file,
        image_category: 'user',
        is_intermediate: false,
        postUploadAction: postUploadAction ?? { type: 'TOAST' },
        board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
      });
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
    multiple: false,
  });

  return { getUploadButtonProps, getUploadInputProps, openUploader };
};
