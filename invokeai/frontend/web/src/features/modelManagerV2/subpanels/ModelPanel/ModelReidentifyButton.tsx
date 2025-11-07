import { Button } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSparkleFill } from 'react-icons/pi';
import { useReidentifyModelMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

interface Props {
  modelConfig: AnyModelConfig;
}

export const ModelReidentifyButton = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const [reidentifyModel, { isLoading }] = useReidentifyModelMutation();

  const onClick = useCallback(() => {
    reidentifyModel({ key: modelConfig.key })
      .unwrap()
      .then(({ type }) => {
        if (type === 'unknown') {
          toast({
            id: 'MODEL_REIDENTIFY_UNKNOWN',
            title: t('modelManager.reidentifyUnknown'),
            status: 'warning',
          });
        }
        toast({
          id: 'MODEL_REIDENTIFY_SUCCESS',
          title: t('modelManager.reidentifySuccess'),
          status: 'success',
        });
      })
      .catch((_) => {
        toast({
          id: 'MODEL_REIDENTIFY_ERROR',
          title: t('modelManager.reidentifyError'),
          status: 'error',
        });
      });
  }, [modelConfig.key, reidentifyModel, t]);

  return (
    <Button
      onClick={onClick}
      size="sm"
      aria-label={t('modelManager.reidentifyTooltip')}
      tooltip={t('modelManager.reidentifyTooltip')}
      isLoading={isLoading}
      flexShrink={0}
      leftIcon={<PiSparkleFill />}
    >
      {t('modelManager.reidentify')}
    </Button>
  );
});

ModelReidentifyButton.displayName = 'ModelReidentifyButton';
