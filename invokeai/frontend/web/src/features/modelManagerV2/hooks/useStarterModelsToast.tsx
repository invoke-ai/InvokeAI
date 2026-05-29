import { Button, Text, useToast } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser, selectIsAuthenticated } from 'features/auth/store/authSlice';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import { useMainModels } from 'services/api/hooks/modelsByType';

const TOAST_ID = 'starterModels';

export const useStarterModelsToast = () => {
  const { t } = useTranslation();
  const [didToast, setDidToast] = useState(false);
  const [mainModels, { data }] = useMainModels();
  const toast = useToast();
  const isAuthenticated = useAppSelector(selectIsAuthenticated);
  const { data: setupStatus } = useGetSetupStatusQuery();
  const user = useAppSelector(selectCurrentUser);

  const isMultiuser = setupStatus?.multiuser_enabled ?? false;
  const isAdmin = !isMultiuser || (user?.is_admin ?? false);

  useEffect(() => {
    // Only show the toast if the user is authenticated
    if (!isAuthenticated) {
      return;
    }

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
        description: isAdmin ? <AdminToastDescription /> : <NonAdminToastDescription />,
        status: 'info',
        isClosable: true,
        duration: null,
        onCloseComplete: () => setDidToast(true),
      });
    }
  }, [data, didToast, isAuthenticated, isAdmin, mainModels.length, t, toast]);
};

const AdminToastDescription = () => {
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

const NonAdminToastDescription = () => {
  const { t } = useTranslation();

  return <Text fontSize="md">{t('modelManager.noModelsInstalledAskAdmin')}</Text>;
};
