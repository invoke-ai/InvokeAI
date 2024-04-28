import { Button, Text, useToast } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useMainModels } from 'services/api/hooks/modelsByType';

const TOAST_ID = 'starterModels';

export const useStarterModelsToast = () => {
  const { t } = useTranslation();
  const isEnabled = useFeatureStatus('starterModels');
  const [didToast, setDidToast] = useState(false);
  const [mainModels, { data }] = useMainModels();
  const toast = useToast();

  useEffect(() => {
    if (toast.isActive(TOAST_ID)) {
      return;
    }
    if (data && mainModels.length === 0 && !didToast && isEnabled) {
      toast({
        id: TOAST_ID,
        title: t('modelManager.noModelsInstalled'),
        description: <ToastDescription />,
        status: 'info',
        isClosable: true,
        duration: null,
        onCloseComplete: () => setDidToast(true),
      });
    }
  }, [data, didToast, isEnabled, mainModels.length, t, toast]);
};

const ToastDescription = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const toast = useToast();

  const onClick = useCallback(() => {
    dispatch(setActiveTab('modelManager'));
    toast.close(TOAST_ID);
  }, [dispatch, toast]);

  return (
    <Text fontSize="md">
      {t('modelManager.noModelsInstalledDesc1')}{' '}
      <Button onClick={onClick} variant="link" color="base.50" flexGrow={0}>
        {t('modelManager.modelManager')}.
      </Button>
    </Text>
  );
};
