import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { InvocationTemplate } from 'features/nodes/types/invocation';

export const useNodeTemplateByType = (type: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }): InvocationTemplate | undefined => {
          const nodeTemplate = nodes.nodeTemplates[type];
          return nodeTemplate;
        },
        defaultSelectorOptions
      ),
    [type]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
