import { Flex, Icon, type SystemStyleObject } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { selectMaxImageUploadCount } from 'features/system/store/configSlice';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import type { FileRejection } from 'react-dropzone';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';
import { uploadImages, useUploadImageMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

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
  const [uploadImage] = useUploadImageMutation();
  const maxImageUploadCount = useAppSelector(selectMaxImageUploadCount);
  const { t } = useTranslation();

  const onDropAccepted = useCallback(
    async (files: File[]) => {
      if (!allowMultiple) {
        if (files.length > 1) {
          log.warn('Multiple files dropped but only one allowed');
          return;
        }
        const file = files[0];
        assert(file !== undefined); // should never happen
        const imageDTO = await uploadImage({
          file,
          image_category: 'user',
          is_intermediate: false,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
        }).unwrap();
        if (onUpload) {
          onUpload(imageDTO);
        }
      } else {
        //
        const imageDTOs = await uploadImages(
          files.map((file) => ({
            file,
            image_category: 'user',
            is_intermediate: false,
            board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
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

  return { getUploadButtonProps, getUploadInputProps, openUploader };
};

const sx = {
  w: 'full',
  h: 'full',
  alignItems: 'center',
  justifyContent: 'center',
  borderRadius: 'base',
  transitionProperty: 'common',
  transitionDuration: '0.1s',
  '&[data-disabled=false]': {
    color: 'base.500',
  },
  '&[data-disabled=true]': {
    cursor: 'pointer',
    bg: 'base.700',
    _hover: {
      bg: 'base.650',
      color: 'base.300',
    },
  },
} satisfies SystemStyleObject;

export const UploadImageButton = (props: UseImageUploadButtonArgs) => {
  const uploadApi = useImageUploadButton(props);
  return (
    <Flex sx={sx} {...uploadApi.getUploadButtonProps()}>
      <Icon as={PiUploadSimpleBold} boxSize={16} />
      <input {...uploadApi.getUploadInputProps()} />
    </Flex>
  );
};
