import { useStore } from '@nanostores/react';
import { useNodeType } from 'features/nodes/hooks/useNodeType';
import { $templates } from 'features/nodes/store/nodesSlice';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useNodeTemplateOrThrow = (nodeId: string): InvocationTemplate => {
  const templates = useStore($templates);
  const type = useNodeType(nodeId);
  const template = useMemo(() => {
    const t = templates[type];
    assert(t, `Template for node type ${type} not found`);
    return t;
  }, [templates, type]);
  return template;
};
