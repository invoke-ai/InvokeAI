import type { ButtonProps, IconButtonProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Button, IconButton } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { trackAsyncTask } from 'common/util/trackAsyncTask';
import { getUploadDropzoneAccept, partitionUploadFiles } from 'common/util/uploadMediaAccept';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useRef, useState } from 'react';
import type { FileRejection } from 'react-dropzone';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadBold } from 'react-icons/pi';
import { uploadImages, useUploadImageMutation } from 'services/api/endpoints/images';
import { uploadVideos, useUploadVideoMutation } from 'services/api/endpoints/videos';
import type { ImageDTO, VideoDTO } from 'services/api/types';
import { assert } from 'tsafe';
import type { SetOptional } from 'type-fest';

type UseImageUploadButtonArgs =
  | {
      isDisabled?: boolean;
      allowMultiple: false;
      /**
       * Opt-in for video uploads. The backend only stores videos in the gallery, so a
       * consumer that wants an image (board covers, ref images, etc.) must NOT set this —
       * otherwise a selected MP4 would be uploaded to the gallery and its DTO silently
       * discarded while the requested image action goes nowhere.
       */
      allowVideos?: boolean;
      onUpload?: (imageDTO: ImageDTO) => void;
      /** Called when a single dropped file is a video (parallel to onUpload for images). */
      onUploadVideo?: (videoDTO: VideoDTO) => void;
      onUploadStarted?: (files: File) => void;
      onError?: (error: unknown) => void;
    }
  | {
      isDisabled?: boolean;
      allowMultiple: true;
      allowVideos?: boolean;
      onUpload?: (imageDTOs: ImageDTO[]) => void;
      onUploadVideo?: (videoDTOs: VideoDTO[]) => void;
      onUploadStarted?: (files: File[]) => void;
      onError?: (error: unknown) => void;
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
export const useImageUploadButton = ({
  onUpload,
  onUploadVideo,
  isDisabled,
  allowMultiple,
  allowVideos = false,
  onUploadStarted,
  onError,
}: UseImageUploadButtonArgs) => {
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const [uploadImage, imageRequest] = useUploadImageMutation();
  const [uploadVideo, videoRequest] = useUploadVideoMutation();
  const [isBatchUploading, setIsBatchUploading] = useState(false);
  const pendingBatchUploads = useRef(0);
  const { t } = useTranslation();
  const onBatchLoadingChanged = useCallback((isLoading: boolean) => {
    pendingBatchUploads.current += isLoading ? 1 : -1;
    setIsBatchUploading(pendingBatchUploads.current > 0);
  }, []);

  const onDropAccepted = useCallback(
    async (files: File[]) => {
      // The accept map already excludes videos for image-only consumers, but the file
      // dialog can bypass it (e.g. "All Files"), so partition again at runtime.
      const { imageFiles, videoFiles, rejectedFiles } = partitionUploadFiles(files, allowVideos);
      if (rejectedFiles.length > 0) {
        log.error({ files: rejectedFiles.map((f) => f.name) }, 'Videos are not accepted by this upload field');
        toast({
          id: 'UPLOAD_FAILED',
          title: t('toast.uploadFailed'),
          description: t('toast.uploadFailedInvalidUploadDesc'),
          status: 'error',
        });
        return;
      }
      try {
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
          onUploadStarted?.(file);

          if (videoFiles.length > 0) {
            const videoDTO = await uploadVideo({
              file,
              video_category: 'user',
              is_intermediate: false,
              board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
              silent: true,
            }).unwrap();
            // Cast: TS narrows onUploadVideo by the allowMultiple discriminator above.
            (onUploadVideo as ((dto: VideoDTO) => void) | undefined)?.(videoDTO);
          } else {
            const imageDTO = await uploadImage({
              file,
              image_category: 'user',
              is_intermediate: false,
              board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
              silent: true,
            }).unwrap();
            (onUpload as ((dto: ImageDTO) => void) | undefined)?.(imageDTO);
          }
        } else {
          onUploadStarted?.(files);
          await trackAsyncTask(async () => {
            let imageDTOs: ImageDTO[] = [];
            if (imageFiles.length > 0) {
              imageDTOs = await uploadImages(
                imageFiles.map((file, i) => ({
                  file,
                  image_category: 'user',
                  is_intermediate: false,
                  board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
                  silent: false,
                  isFirstUploadOfBatch: i === 0,
                }))
              );
            }

            let videoDTOs: VideoDTO[] = [];
            if (videoFiles.length > 0) {
              videoDTOs = await uploadVideos(
                videoFiles.map((file, i) => ({
                  file,
                  video_category: 'user',
                  is_intermediate: false,
                  board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
                  silent: false,
                  isFirstUploadOfBatch: i === 0,
                }))
              );
            }

            if (imageDTOs.length > 0) {
              (onUpload as ((dtos: ImageDTO[]) => void) | undefined)?.(imageDTOs);
            }
            if (videoDTOs.length > 0) {
              (onUploadVideo as ((dtos: VideoDTO[]) => void) | undefined)?.(videoDTOs);
            }
          }, onBatchLoadingChanged);
        }
      } catch (error) {
        onError?.(error);
        // Name the media that actually failed — a failed MP4 upload should not claim an
        // image upload failed. Mixed batches get the media-neutral title.
        const title =
          videoFiles.length > 0 && imageFiles.length > 0
            ? t('toast.uploadFailed')
            : videoFiles.length > 0
              ? t('toast.videoUploadFailed')
              : t('toast.imageUploadFailed');
        toast({
          id: 'UPLOAD_FAILED',
          title,
          status: 'error',
        });
      }
    },
    [
      allowMultiple,
      allowVideos,
      onUploadStarted,
      uploadImage,
      uploadVideo,
      autoAddBoardId,
      onUpload,
      onUploadVideo,
      onError,
      onBatchLoadingChanged,
      t,
    ]
  );

  const onDropRejected = useCallback(
    (fileRejections: FileRejection[]) => {
      if (fileRejections.length > 0) {
        const errors = fileRejections.map((rejection) => ({
          errors: rejection.errors.map(({ message }) => message),
          file: rejection.file.path,
        }));
        log.error({ errors }, 'Invalid upload');
        const description = t('toast.uploadFailedInvalidUploadDesc');

        toast({
          id: 'UPLOAD_FAILED',
          title: t('toast.uploadFailed'),
          description,
          status: 'error',
        });

        return;
      }
    },
    [t]
  );

  const {
    getRootProps: getUploadButtonProps,
    getInputProps: getUploadInputProps,
    open: openUploader,
  } = useDropzone({
    accept: getUploadDropzoneAccept(allowVideos),
    onDropAccepted,
    onDropRejected,
    disabled: isDisabled,
    noDrag: true,
    multiple: allowMultiple,
  });

  // Uploads run through separate image and video mutations; loading state must cover both
  // or an in-flight MP4 upload would show idle controls and allow double submission.
  const isUploading = imageRequest.isLoading || videoRequest.isLoading || isBatchUploading;

  return { getUploadButtonProps, getUploadInputProps, openUploader, isUploading };
};

const sx = {
  '&[data-error=true]': {
    borderColor: 'error.500',
    borderStyle: 'solid',
    borderWidth: 1,
  },
} satisfies SystemStyleObject;

export const UploadImageIconButton = memo(
  ({
    isDisabled = false,
    onUpload,
    isError = false,
    ...rest
  }: {
    onUpload?: (imageDTO: ImageDTO) => void;
    isError?: boolean;
  } & SetOptional<IconButtonProps, 'aria-label'>) => {
    const { t } = useTranslation();
    const uploadApi = useImageUploadButton({ isDisabled, allowMultiple: false, onUpload });
    return (
      <>
        <IconButton
          aria-label={t('accessibility.uploadImage')}
          variant="outline"
          sx={sx}
          data-error={isError}
          icon={<PiUploadBold />}
          isLoading={uploadApi.isUploading}
          {...rest}
          {...uploadApi.getUploadButtonProps()}
        />
        <input {...uploadApi.getUploadInputProps()} />
      </>
    );
  }
);
UploadImageIconButton.displayName = 'UploadImageIconButton';

type UploadImageButtonProps = {
  onUpload?: (imageDTO: ImageDTO) => void;
  isError?: boolean;
} & ButtonProps;

const UploadImageButton = memo((props: UploadImageButtonProps) => {
  const { t } = useTranslation();
  const { children, isDisabled = false, onUpload, isError = false, ...rest } = props;
  const uploadApi = useImageUploadButton({ isDisabled, allowMultiple: false, onUpload });
  return (
    <>
      <Button
        aria-label={t('accessibility.uploadImage')}
        variant="outline"
        sx={sx}
        data-error={isError}
        rightIcon={<PiUploadBold />}
        isLoading={uploadApi.isUploading}
        {...rest}
        {...uploadApi.getUploadButtonProps()}
      >
        {children ?? 'Upload'}
      </Button>
      <input {...uploadApi.getUploadInputProps()} />
    </>
  );
});
UploadImageButton.displayName = 'UploadImageButton';

export const UploadMultipleImageButton = ({
  isDisabled = false,
  onUpload,
  isError = false,
  ...rest
}: {
  onUpload?: (imageDTOs: ImageDTO[]) => void;
  isError?: boolean;
} & SetOptional<IconButtonProps, 'aria-label'>) => {
  const { t } = useTranslation();
  const uploadApi = useImageUploadButton({ isDisabled, allowMultiple: true, onUpload });
  return (
    <>
      <IconButton
        aria-label={t('accessibility.uploadImage')}
        variant="outline"
        sx={sx}
        data-error={isError}
        icon={<PiUploadBold />}
        isLoading={uploadApi.isUploading}
        {...rest}
        {...uploadApi.getUploadButtonProps()}
      />
      <input {...uploadApi.getUploadInputProps()} />
    </>
  );
};
