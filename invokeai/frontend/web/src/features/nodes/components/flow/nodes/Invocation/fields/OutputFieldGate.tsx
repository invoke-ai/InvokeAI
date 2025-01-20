import { OutputFieldUnknownPlaceholder } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldUnknownPlaceholder';
import { useOutputFieldTemplateExists } from 'features/nodes/hooks/useOutputFieldTemplateExists';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
}>;

export const OutputFieldGate = memo(({ nodeId, fieldName, children }: Props) => {
  const hasTemplate = useOutputFieldTemplateExists(nodeId, fieldName);

  if (!hasTemplate) {
    return <OutputFieldUnknownPlaceholder nodeId={nodeId} fieldName={fieldName} />;
  }

  return children;
});

OutputFieldGate.displayName = 'OutputFieldGate';
