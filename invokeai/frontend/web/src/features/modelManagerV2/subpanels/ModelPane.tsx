import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo } from 'react';

import { InstallModels } from './InstallModels';
import { Model } from './ModelPanel/Model';

const modelPaneSx: SystemStyleObject = {
  layerStyle: 'first',
  p: 4,
  borderRadius: 'base',
  w: {
    base: '50%',
    lg: '75%',
    '2xl': '85%',
  },
  h: 'full',
  minWidth: '300px',
};

export const ModelPane = memo(() => {
  const selectedModelKey = useAppSelector(selectSelectedModelKey);
  return <Box sx={modelPaneSx}>{selectedModelKey ? <Model key={selectedModelKey} /> : <InstallModels />}</Box>;
});

ModelPane.displayName = 'ModelPane';
