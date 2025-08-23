import { useStore } from '@nanostores/react';
import { $authToken } from 'app/store/nanostores/authToken';
import { useAppDispatch } from 'app/store/storeHooks';
import { imageDownloaded } from 'features/gallery/store/actions';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const useDownloadItem = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const authToken = useStore($authToken);

  const downloadItem = useCallback(
    async (item_url: string, item_id: string) => {
      try {
        const requestOpts = authToken
          ? {
              headers: {
                Authorization: `Bearer ${authToken}`,
              },
            }
          : {};
        const blob = await fetch(item_url, requestOpts).then((resp) => resp.blob());
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
        dispatch(imageDownloaded());
      } catch (err) {
        toast({
          id: 'PROBLEM_DOWNLOADING_IMAGE',
          title: t('toast.problemDownloadingImage'),
          description: String(err),
          status: 'error',
        });
      }
    },
    [t, dispatch, authToken]
  );

  return { downloadItem };
};
