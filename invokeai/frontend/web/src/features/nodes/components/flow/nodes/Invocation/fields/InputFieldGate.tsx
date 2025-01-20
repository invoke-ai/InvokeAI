import { InputFieldUnknownPlaceholder } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldUnknownPlaceholder';
import { useFieldInputInstanceExists } from 'features/nodes/hooks/useFieldInputInstanceExists';
import { useFieldInputTemplateExists } from 'features/nodes/hooks/useFieldInputTemplateExists';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
}>;

export const InputFieldGate = memo(({ nodeId, fieldName, children }: Props) => {
  const hasInstance = useFieldInputInstanceExists(nodeId, fieldName);
  const hasTemplate = useFieldInputTemplateExists(nodeId, fieldName);

  if (!hasTemplate || !hasInstance) {
    return <InputFieldUnknownPlaceholder nodeId={nodeId} fieldName={fieldName} />;
  }

  return children;
});

InputFieldGate.displayName = 'InputFieldGate';
