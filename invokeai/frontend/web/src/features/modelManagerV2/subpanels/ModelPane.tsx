import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';

import { InstallModels } from './InstallModels';
import { Model } from './ModelPanel/Model';

export const ModelPane = memo(() => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  if (selectedModelKey) {
    return <Model key={selectedModelKey} />;
  }

  return <InstallModels />;
});

ModelPane.displayName = 'ModelPane';
