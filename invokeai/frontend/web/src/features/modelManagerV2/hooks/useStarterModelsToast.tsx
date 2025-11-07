import { Button, Text, useToast } from '@invoke-ai/ui-library';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useMainModels } from 'services/api/hooks/modelsByType';

const TOAST_ID = 'starterModels';

export const useStarterModelsToast = () => {
  const { t } = useTranslation();
  const [didToast, setDidToast] = useState(false);
  const [mainModels, { data }] = useMainModels();
  const toast = useToast();

  useEffect(() => {
    if (toast.isActive(TOAST_ID)) {
      if (mainModels.length === 0) {
        return;
      } else {
        toast.close(TOAST_ID);
      }
    }
    if (data && mainModels.length === 0 && !didToast) {
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
  }, [data, didToast, mainModels.length, t, toast]);
};

const ToastDescription = () => {
  const { t } = useTranslation();
  const toast = useToast();

  const onClick = useCallback(() => {
    navigationApi.switchToTab('models');
    setInstallModelsTabByName('launchpad');
    toast.close(TOAST_ID);
  }, [toast]);

  return (
    <Text fontSize="md">
      {t('modelManager.noModelsInstalledDesc1')}{' '}
      <Button onClick={onClick} variant="link" color="base.50" flexGrow={0}>
        {t('ui.tabs.modelsTab')}.
      </Button>
    </Text>
  );
};
