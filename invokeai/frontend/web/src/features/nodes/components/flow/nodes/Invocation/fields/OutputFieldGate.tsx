import { OutputFieldUnknownPlaceholder } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldUnknownPlaceholder';
import { useFieldOutputTemplateExists } from 'features/nodes/hooks/useFieldOutputTemplateExists';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
}>;

export const OutputFieldGate = memo(({ nodeId, fieldName, children }: Props) => {
  const hasTemplate = useFieldOutputTemplateExists(nodeId, fieldName);

  if (!hasTemplate) {
    return <OutputFieldUnknownPlaceholder nodeId={nodeId} fieldName={fieldName} />;
  }

  return children;
});

OutputFieldGate.displayName = 'OutputFieldGate';
