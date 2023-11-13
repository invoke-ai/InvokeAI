import { useAppToaster } from 'app/components/Toaster';
import { useImageUrlToBlob } from 'common/hooks/useImageUrlToBlob';
import { copyBlobToClipboard } from 'features/system/util/copyBlobToClipboard';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const useCopyImageToClipboard = () => {
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const imageUrlToBlob = useImageUrlToBlob();

  const isClipboardAPIAvailable = useMemo(() => {
    return Boolean(navigator.clipboard) && Boolean(window.ClipboardItem);
  }, []);

  const copyImageToClipboard = useCallback(
    async (image_url: string) => {
      if (!isClipboardAPIAvailable) {
        toaster({
          title: t('toast.problemCopyingImage'),
          description: "Your browser doesn't support the Clipboard API.",
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
      try {
        const blob = await imageUrlToBlob(image_url);

        if (!blob) {
          throw new Error('Unable to create Blob');
        }

        copyBlobToClipboard(blob);

        toaster({
          title: t('toast.imageCopied'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } catch (err) {
        toaster({
          title: t('toast.problemCopyingImage'),
          description: String(err),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [imageUrlToBlob, isClipboardAPIAvailable, t, toaster]
  );

  return { isClipboardAPIAvailable, copyImageToClipboard };
};
