import { InputFieldUnknownPlaceholder } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldUnknownPlaceholder';
import { useInputFieldInstanceExists } from 'features/nodes/hooks/useInputFieldInstanceExists';
import { useInputFieldTemplateExists } from 'features/nodes/hooks/useInputFieldTemplateExists';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
}>;

export const InputFieldGate = memo(({ nodeId, fieldName, children }: Props) => {
  const hasInstance = useInputFieldInstanceExists(nodeId, fieldName);
  const hasTemplate = useInputFieldTemplateExists(nodeId, fieldName);

  if (!hasTemplate || !hasInstance) {
    return <InputFieldUnknownPlaceholder nodeId={nodeId} fieldName={fieldName} />;
  }

  return children;
});

InputFieldGate.displayName = 'InputFieldGate';
