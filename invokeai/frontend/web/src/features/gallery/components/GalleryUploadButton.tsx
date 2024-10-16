import { IconButton } from '@invoke-ai/ui-library';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { t } from 'i18next';
import { PiUploadBold } from 'react-icons/pi';

export const GalleryUploadButton = () => {
  const uploadApi = useImageUploadButton({ postUploadAction: { type: 'TOAST' } });
  return (
    <>
      <IconButton
        size="sm"
        alignSelf="stretch"
        variant="link"
        aria-label={t('accessibility.uploadImage')}
        tooltip={t('accessibility.uploadImage')}
        icon={<PiUploadBold />}
        {...uploadApi.getUploadButtonProps()}
      />
      <input {...uploadApi.getUploadInputProps()} />
    </>
  );
};
