import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { type InstallModelArg, useInstallModelMutation } from 'services/api/endpoints/models';

type InstallModelArgWithCallbacks = InstallModelArg & {
  onSuccess?: () => void;
  onError?: (error: unknown) => void;
};

export const useInstallModel = () => {
  const { t } = useTranslation();
  const [_installModel, request] = useInstallModelMutation();

  const installModel = useCallback(
    ({ source, inplace, config, onSuccess, onError }: InstallModelArgWithCallbacks) => {
      config ||= {};
      _installModel({ source, inplace, config })
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
