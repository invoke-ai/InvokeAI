import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeTemplateByType = (type: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(
        stateSelector,
        ({ nodeTemplates }): InvocationTemplate | undefined => {
          return nodeTemplates.templates[type];
        }
      ),
    [type]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
