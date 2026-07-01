import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const useDownloadItem = () => {
  const { t } = useTranslation();

  const downloadItem = useCallback(
    async (item_url: string, item_id: string) => {
      try {
        const blob = await fetch(item_url).then((resp) => resp.blob());
        if (!blob) {
          throw new Error('Unable to create Blob');
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = item_id;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
      } catch (err) {
        toast({
          id: 'PROBLEM_DOWNLOADING_IMAGE',
          title: t('toast.problemDownloadingImage'),
          description: String(err),
          status: 'error',
        });
      }
    },
    [t]
  );

  return { downloadItem };
};
