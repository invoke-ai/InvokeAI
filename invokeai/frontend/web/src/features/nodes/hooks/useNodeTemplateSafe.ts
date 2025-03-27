import { useStore } from '@nanostores/react';
import { useNodeType } from 'features/nodes/hooks/useNodeType';
import { $templates } from 'features/nodes/store/nodesSlice';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeTemplateSafe = (nodeId: string): InvocationTemplate | null => {
  const templates = useStore($templates);
  const type = useNodeType(nodeId);
  const template = useMemo(() => templates[type] ?? null, [templates, type]);
  return template;
};
