import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { useMemo } from 'react';

export const useNodeTemplateTitle = (nodeId: string): string | null => {
  const template = useNodeTemplate(nodeId);
  const title = useMemo(() => template.title, [template.title]);
  return title;
};
