import { useMemo } from 'react';
import type { S } from 'services/api/types';

import { useStarterBundleInstall } from './useStarterBundleInstall';

export const useStarterBundleInstallStatus = (bundle: S['StarterModelBundle']) => {
  const { getModelsToInstall } = useStarterBundleInstall();
  const total = useMemo(() => bundle.models.length, [bundle.models.length]);
  const install = useMemo(() => getModelsToInstall(bundle).install, [bundle, getModelsToInstall]);
  const skip = useMemo(() => getModelsToInstall(bundle).skip, [bundle, getModelsToInstall]);

  return { total, skip, install };
};
