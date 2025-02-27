import type { IconButtonProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { selectMaxImageUploadCount } from 'features/system/store/configSlice';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import type { FileRejection } from 'react-dropzone';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadBold } from 'react-icons/pi';
import { uploadImages, useUploadImageMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';
import type { SetOptional } from 'type-fest';

type UseImageUploadButtonArgs =
  | {
      isDisabled?: boolean;
      allowMultiple: false;
      onUpload?: (imageDTO: ImageDTO) => void;
    }
  | {
      isDisabled?: boolean;
      allowMultiple: true;
      onUpload?: (imageDTOs: ImageDTO[]) => void;
    };

const log = logger('gallery');

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
export const useImageUploadButton = ({ onUpload, isDisabled, allowMultiple }: UseImageUploadButtonArgs) => {
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const [uploadImage, request] = useUploadImageMutation();
  const maxImageUploadCount = useAppSelector(selectMaxImageUploadCount);
  const { t } = useTranslation();

  const onDropAccepted = useCallback(
    async (files: File[]) => {
      if (!allowMultiple) {
        if (files.length > 1) {
          log.warn('Multiple files dropped but only one allowed');
          return;
        }
        if (files.length === 0) {
          // Should never happen
          log.warn('No files dropped');
          return;
        }
        const file = files[0];
        assert(file !== undefined); // should never happen
        const imageDTO = await uploadImage({
          file,
          image_category: 'user',
          is_intermediate: false,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
          silent: true,
        }).unwrap();
        if (onUpload) {
          onUpload(imageDTO);
        }
      } else {
        const imageDTOs = await uploadImages(
          files.map((file, i) => ({
            file,
            image_category: 'user',
            is_intermediate: false,
            board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
            silent: false,
            isFirstUploadOfBatch: i === 0,
          }))
        );
        if (onUpload) {
          onUpload(imageDTOs);
        }
      }
    },
    [allowMultiple, autoAddBoardId, onUpload, uploadImage]
  );

  const onDropRejected = useCallback(
    (fileRejections: FileRejection[]) => {
      if (fileRejections.length > 0) {
        const errors = fileRejections.map((rejection) => ({
          errors: rejection.errors.map(({ message }) => message),
          file: rejection.file.path,
        }));
        log.error({ errors }, 'Invalid upload');
        const description =
          maxImageUploadCount === undefined
            ? t('toast.uploadFailedInvalidUploadDesc')
            : t('toast.uploadFailedInvalidUploadDesc_withCount', { count: maxImageUploadCount });

        toast({
          id: 'UPLOAD_FAILED',
          title: t('toast.uploadFailed'),
          description,
          status: 'error',
        });

        return;
      }
    },
    [maxImageUploadCount, t]
  );

  const {
    getRootProps: getUploadButtonProps,
    getInputProps: getUploadInputProps,
    open: openUploader,
  } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    onDropAccepted,
    onDropRejected,
    disabled: isDisabled,
    noDrag: true,
    multiple: allowMultiple && (maxImageUploadCount === undefined || maxImageUploadCount > 1),
    maxFiles: maxImageUploadCount,
  });

  return { getUploadButtonProps, getUploadInputProps, openUploader, request };
};

const sx = {
  '&[data-error=true]': {
    borderColor: 'error.500',
    borderStyle: 'solid',
    borderWidth: 1,
  },
} satisfies SystemStyleObject;

export const UploadImageButton = ({
  isDisabled = false,
  onUpload,
  isError = false,
  ...rest
}: {
  onUpload?: (imageDTO: ImageDTO) => void;
  isError?: boolean;
} & SetOptional<IconButtonProps, 'aria-label'>) => {
  const uploadApi = useImageUploadButton({ isDisabled, allowMultiple: false, onUpload });
  return (
    <>
      <IconButton
        aria-label="Upload image"
        variant="outline"
        sx={sx}
        data-error={isError}
        icon={<PiUploadBold />}
        isLoading={uploadApi.request.isLoading}
        {...rest}
        {...uploadApi.getUploadButtonProps()}
      />
      <input {...uploadApi.getUploadInputProps()} />
    </>
  );
};

export const UploadMultipleImageButton = ({
  isDisabled = false,
  onUpload,
  isError = false,
  ...rest
}: {
  onUpload?: (imageDTOs: ImageDTO[]) => void;
  isError?: boolean;
} & SetOptional<IconButtonProps, 'aria-label'>) => {
  const uploadApi = useImageUploadButton({ isDisabled, allowMultiple: true, onUpload });
  return (
    <>
      <IconButton
        aria-label="Upload image"
        variant="outline"
        sx={sx}
        data-error={isError}
        icon={<PiUploadBold />}
        isLoading={uploadApi.request.isLoading}
        {...rest}
        {...uploadApi.getUploadButtonProps()}
      />
      <input {...uploadApi.getUploadInputProps()} />
    </>
  );
};
