import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const useDownloadItem = () => {
  const { t } = useTranslation();

  const downloadItem = useCallback(
    async (item_url: string, item_id: string) => {
      try {
        const resp = await fetch(item_url);
        if (!resp.ok) {
          // Without this check an error response body (e.g. a 401 after the media cookie
          // expired) would be saved as the media file itself.
          throw new Error(`Server returned ${resp.status}`);
        }
        const blob = await resp.blob();
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
