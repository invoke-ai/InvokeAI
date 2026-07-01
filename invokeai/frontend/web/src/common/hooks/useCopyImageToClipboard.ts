import { useClipboard } from 'common/hooks/useClipboard';
import { convertImageUrlToBlob } from 'common/util/convertImageUrlToBlob';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const useCopyImageToClipboard = () => {
  const { t } = useTranslation();
  const clipboard = useClipboard();

  const copyImageToClipboard = useCallback(
    async (image_url: string) => {
      try {
        const blob = await convertImageUrlToBlob(image_url);

        if (!blob) {
          throw new Error('Unable to create Blob');
        }

        clipboard.writeImage(blob, () => {
          toast({
            id: 'IMAGE_COPIED',
            title: t('toast.imageCopied'),
            status: 'success',
          });
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
    [clipboard, t]
  );

  return copyImageToClipboard;
};
