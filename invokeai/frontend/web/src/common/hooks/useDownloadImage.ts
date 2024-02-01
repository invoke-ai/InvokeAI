import { useAppToaster } from 'app/components/Toaster';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const useDownloadImage = () => {
  const toaster = useAppToaster();
  const { t } = useTranslation();

  const downloadImage = useCallback(
    async (image_url: string, image_name: string) => {
      try {
        const blob = await fetch(image_url).then((resp) => resp.blob());

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = image_name;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
      } catch (err) {
        toaster({
          title: t('toast.problemDownloadingImage'),
          description: String(err),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [t, toaster]
  );

  return { downloadImage };
};
