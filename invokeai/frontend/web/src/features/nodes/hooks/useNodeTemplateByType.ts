import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvocationTemplate } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeTemplateByType = (type: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(
        stateSelector,
        ({ nodes }): InvocationTemplate | undefined => {
          const nodeTemplate = nodes.nodeTemplates[type];
          return nodeTemplate;
        }
      ),
    [type]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
