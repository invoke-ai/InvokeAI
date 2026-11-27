import { flatMap, negate, uniqWith } from 'es-toolkit/compat';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useInstallModelMutation } from 'services/api/endpoints/models';
import type { S } from 'services/api/types';

import { flattenStarterModel, useBuildModelInstallArg } from './useBuildModelsToInstall';

export const useStarterBundleInstall = () => {
  const [installModel] = useInstallModelMutation();
  const { getIsInstalled, buildModelInstallArg } = useBuildModelInstallArg();
  const { t } = useTranslation();

  const getModelsToInstall = useCallback(
    (bundle: S['StarterModelBundle']) => {
      // Flatten the models and remove duplicates, which is expected as models can have the same dependencies
      const flattenedModels = flatMap(bundle.models, flattenStarterModel);
      const uniqueModels = uniqWith(
        flattenedModels,
        (m1, m2) => m1.source === m2.source || (m1.name === m2.name && m1.base === m2.base && m1.type === m2.type)
      );
      // We want to install models that are not installed and skip models that are already installed
      const install = uniqueModels.filter(negate(getIsInstalled)).map(buildModelInstallArg);
      const skip = uniqueModels.filter(getIsInstalled).map(buildModelInstallArg);

      return { install, skip };
    },
    [getIsInstalled, buildModelInstallArg]
  );

  const installBundle = useCallback(
    (bundle: S['StarterModelBundle']) => {
      const modelsToInstall = getModelsToInstall(bundle);

      if (modelsToInstall.install.length === 0) {
        toast({
          status: 'info',
          title: t('modelManager.bundleAlreadyInstalled'),
          description: t('modelManager.bundleAlreadyInstalledDesc', { bundleName: bundle.name }),
        });
        return;
      }

      // Install all models in the bundle
      modelsToInstall.install.forEach(installModel);

      let description = t('modelManager.installingXModels', { count: modelsToInstall.install.length });
      if (modelsToInstall.skip.length > 1) {
        description += t('modelManager.skippingXDuplicates', { count: modelsToInstall.skip.length - 1 });
      }

      toast({
        status: 'info',
        title: t('modelManager.installingBundle'),
        description,
      });
    },
    [getModelsToInstall, installModel, t]
  );

  return { installBundle, getModelsToInstall };
};
