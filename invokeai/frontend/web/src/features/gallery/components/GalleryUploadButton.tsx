import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { selectMaxImageUploadCount } from 'features/system/store/configSlice';
import { t } from 'i18next';
import { PiUploadBold } from 'react-icons/pi';

const options = { postUploadAction: { type: 'TOAST' }, allowMultiple: true } as const;

export const GalleryUploadButton = () => {
  const uploadApi = useImageUploadButton(options);
  const maxImageUploadCount = useAppSelector(selectMaxImageUploadCount);
  return (
    <>
      <IconButton
        size="sm"
        alignSelf="stretch"
        variant="link"
        aria-label={
          maxImageUploadCount === undefined || maxImageUploadCount > 1
            ? t('accessibility.uploadImages')
            : t('accessibility.uploadImage')
        }
        tooltip={
          maxImageUploadCount === undefined || maxImageUploadCount > 1
            ? t('accessibility.uploadImages')
            : t('accessibility.uploadImage')
        }
        icon={<PiUploadBold />}
        {...uploadApi.getUploadButtonProps()}
      />
      <input {...uploadApi.getUploadInputProps()} />
    </>
  );
};
