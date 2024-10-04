import { convertImageUrlToBlob } from 'common/util/convertImageUrlToBlob';
import { copyBlobToClipboard } from 'features/system/util/copyBlobToClipboard';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const useCopyImageToClipboard = () => {
  const { t } = useTranslation();

  const isClipboardAPIAvailable = useMemo(() => {
    return Boolean(navigator.clipboard) && Boolean(window.ClipboardItem);
  }, []);

  const copyImageToClipboard = useCallback(
    async (image_url: string) => {
      if (!isClipboardAPIAvailable) {
        toast({
          id: 'PROBLEM_COPYING_IMAGE',
          title: t('toast.problemCopyingImage'),
          description: "Your browser doesn't support the Clipboard API.",
          status: 'error',
        });
      }
      try {
        const blob = await convertImageUrlToBlob(image_url);

        if (!blob) {
          throw new Error('Unable to create Blob');
        }

        copyBlobToClipboard(blob);

        toast({
          id: 'IMAGE_COPIED',
          title: t('toast.imageCopied'),
          status: 'success',
        });
      } catch (err) {
        toast({
          id: 'PROBLEM_COPYING_IMAGE',
          title: t('toast.problemCopyingImage'),
          description: String(err),
          status: 'error',
        });
      }
    },
    [isClipboardAPIAvailable, t]
  );

  return { isClipboardAPIAvailable, copyImageToClipboard };
};
