import { IconButton } from '@invoke-ai/ui-library';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { t } from 'i18next';
import { memo } from 'react';
import { PiUploadBold } from 'react-icons/pi';

// The gallery uploader is the one place videos are accepted — every other consumer wants an image.
const UPLOAD_OPTIONS: Parameters<typeof useImageUploadButton>[0] = { allowMultiple: true, allowVideos: true };

export const GalleryUploadButton = memo(() => {
  const uploadApi = useImageUploadButton(UPLOAD_OPTIONS);
  return (
    <>
      <IconButton
        size="sm"
        alignSelf="stretch"
        variant="link"
        aria-label={t('accessibility.uploadImages')}
        tooltip={t('accessibility.uploadImages')}
        icon={<PiUploadBold />}
        {...uploadApi.getUploadButtonProps()}
      />
      <input {...uploadApi.getUploadInputProps()} />
    </>
  );
});
GalleryUploadButton.displayName = 'GalleryUploadButton';
