import { InputFieldUnknownPlaceholder } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldUnknownPlaceholder';
import { useInputFieldInstanceExists } from 'features/nodes/hooks/useInputFieldInstanceExists';
import { useInputFieldTemplateExists } from 'features/nodes/hooks/useInputFieldTemplateExists';
import type { PropsWithChildren, ReactNode } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
  placeholder?: ReactNode;
}>;

export const InputFieldGate = memo(({ nodeId, fieldName, children, placeholder }: Props) => {
  const hasInstance = useInputFieldInstanceExists(nodeId, fieldName);
  const hasTemplate = useInputFieldTemplateExists(nodeId, fieldName);

  if (!hasTemplate || !hasInstance) {
    return placeholder ?? <InputFieldUnknownPlaceholder nodeId={nodeId} fieldName={fieldName} />;
  }

  return children;
});

InputFieldGate.displayName = 'InputFieldGate';
