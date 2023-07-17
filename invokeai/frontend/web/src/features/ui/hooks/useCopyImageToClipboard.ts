import { useAppToaster } from 'app/components/Toaster';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const useCopyImageToClipboard = () => {
  const toaster = useAppToaster();
  const { t } = useTranslation();

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
        const response = await fetch(image_url);
        const blob = await response.blob();
        await navigator.clipboard.write([
          new ClipboardItem({
            [blob.type]: blob,
          }),
        ]);
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
    [isClipboardAPIAvailable, t, toaster]
  );

  return { isClipboardAPIAvailable, copyImageToClipboard };
};
