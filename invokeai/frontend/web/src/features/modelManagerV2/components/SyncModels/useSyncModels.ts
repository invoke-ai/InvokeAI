import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useSyncModelsMutation } from 'services/api/endpoints/models';

export const useSyncModels = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [_syncModels, { isLoading }] = useSyncModelsMutation();
  const syncModels = useCallback(() => {
    _syncModels()
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: `${t('modelManager.modelsSynced')}`,
              status: 'success',
            })
          )
        );
      })
      .catch((error) => {
        if (error) {
          dispatch(
            addToast(
              makeToast({
                title: `${t('modelManager.modelSyncFailed')}`,
                status: 'error',
              })
            )
          );
        }
      });
  }, [dispatch, _syncModels, t]);

  return { syncModels, isLoading };
};
