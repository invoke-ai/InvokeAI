import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useInstallModelMutation } from 'services/api/endpoints/models';

type InstallModelArg = {
  source: string;
  inplace?: boolean;
  onSuccess?: () => void;
  onError?: (error: unknown) => void;
};

export const useInstallModel = () => {
  const { t } = useTranslation();
  const [_installModel, request] = useInstallModelMutation();

  const installModel = useCallback(
    ({ source, inplace, onSuccess, onError }: InstallModelArg) => {
      _installModel({ source, inplace })
        .unwrap()
        .then((_) => {
          if (onSuccess) {
            onSuccess();
          }
          toast({
            id: 'MODEL_INSTALL_QUEUED',
            title: t('toast.modelAddedSimple'),
            status: 'success',
          });
        })
        .catch((error) => {
          if (onError) {
            onError(error);
          }
          if (error) {
            toast({
              id: 'MODEL_INSTALL_QUEUE_FAILED',
              title: `${error.data.detail} `,
              status: 'error',
            });
          }
        });
    },
    [_installModel, t]
  );

  return [installModel, request] as const;
};
