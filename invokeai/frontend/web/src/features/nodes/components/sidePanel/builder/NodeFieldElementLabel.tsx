import { Flex, FormLabel, Spacer } from '@invoke-ai/ui-library';
import { NodeFieldElementResetToInitialValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/NodeFieldElementResetToInitialValueIconButton';
import { useInputFieldTemplateOrThrow } from 'features/nodes/hooks/useInputFieldTemplateOrThrow';
import { useInputFieldUserTitleSafe } from 'features/nodes/hooks/useInputFieldUserTitleSafe';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { memo, useMemo } from 'react';

export const NodeFieldElementLabel = memo(({ el }: { el: NodeFieldElement }) => {
  const { data } = el;
  const { fieldIdentifier } = data;
  const label = useInputFieldUserTitleSafe(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplateOrThrow(fieldIdentifier.nodeId, fieldIdentifier.fieldName);

  const _label = useMemo(() => label || fieldTemplate.title, [label, fieldTemplate.title]);

  return (
    <Flex w="full" gap={4}>
      <FormLabel>{_label}</FormLabel>
      <Spacer />
      <NodeFieldElementResetToInitialValueIconButton element={el} />
    </Flex>
  );
});
NodeFieldElementLabel.displayName = 'NodeFieldElementLabel';
