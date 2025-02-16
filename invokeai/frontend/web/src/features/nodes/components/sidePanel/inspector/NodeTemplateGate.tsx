import { useNodeTemplateSafe } from 'features/nodes/hooks/useNodeTemplate';
import type { PropsWithChildren, ReactNode } from 'react';
import { memo } from 'react';

export const TemplateGate = memo(
  ({ nodeId, fallback, children }: PropsWithChildren<{ nodeId: string; fallback: ReactNode }>) => {
    const template = useNodeTemplateSafe(nodeId);

    if (!template) {
      return fallback;
    }

    return <>{children}</>;
  }
);
TemplateGate.displayName = 'TemplateGate';
