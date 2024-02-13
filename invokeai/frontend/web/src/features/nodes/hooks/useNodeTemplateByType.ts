import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodeTemplatesSlice } from 'features/nodes/store/nodeTemplatesSlice';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeTemplateByType = (type: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodeTemplatesSlice, (nodeTemplates): InvocationTemplate | undefined => {
        return nodeTemplates.templates[type];
      }),
    [type]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
