import { ButtonProps } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useTranslation } from 'react-i18next';
import { FaSync } from 'react-icons/fa';
import { useSyncModelsMutation } from 'services/api/endpoints/models';

type SyncModelsButtonProps = ButtonProps & {
  iconMode?: boolean;
};

export default function SyncModelsButton(props: SyncModelsButtonProps) {
  const { iconMode = false, ...rest } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const [syncModels, { isLoading }] = useSyncModelsMutation();

  const syncModelsHandler = () => {
    syncModels()
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
  };

  return !iconMode ? (
    <IAIButton
      isLoading={isLoading}
      onClick={syncModelsHandler}
      minW="max-content"
      {...rest}
    >
      Sync Models
    </IAIButton>
  ) : (
    <IAIIconButton
      icon={<FaSync />}
      tooltip={t('modelManager.syncModels')}
      aria-label={t('modelManager.syncModels')}
      isLoading={isLoading}
      onClick={syncModelsHandler}
      size="sm"
      {...rest}
    />
  );
}
