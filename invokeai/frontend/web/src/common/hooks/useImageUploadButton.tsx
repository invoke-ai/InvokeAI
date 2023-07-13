import { useAppDispatch } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { PostUploadAction, imageUploaded } from 'services/api/thunks/image';

type UseImageUploadButtonArgs = {
  postUploadAction?: PostUploadAction;
  isDisabled?: boolean;
};

/**
 * Provides image uploader functionality to any component.
 *
 * @example
 * const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
 *   postUploadAction: {
 *     type: 'SET_CONTROLNET_IMAGE',
 *     controlNetId: '12345',
 *   },
 *   isDisabled: getIsUploadDisabled(),
 * });
 *
 * // in the render function
 * <Button {...getUploadButtonProps()} /> // will open the file dialog on click
 * <input {...getUploadInputProps()} /> // hidden, handles native upload functionality
 */
export const useImageUploadButton = ({
  postUploadAction,
  isDisabled,
}: UseImageUploadButtonArgs) => {
  const dispatch = useAppDispatch();
  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (!file) {
        return;
      }

      dispatch(
        imageUploaded({
          file,
          image_category: 'user',
          is_intermediate: false,
          postUploadAction,
        })
      );
    },
    [dispatch, postUploadAction]
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
