import { Flex, FormControl, FormHelperText } from '@invoke-ai/ui-library';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { NodeFieldElementLabel } from 'features/nodes/components/sidePanel/builder/NodeFieldElementLabel';
import { useInputFieldDescription } from 'features/nodes/hooks/useInputFieldDescription';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { NODE_FIELD_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo, useMemo } from 'react';

export const NodeFieldElementViewMode = memo(({ el }: { el: NodeFieldElement }) => {
  const { id, data } = el;
  const { fieldIdentifier, showDescription } = data;
  const description = useInputFieldDescription(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplate(fieldIdentifier.nodeId, fieldIdentifier.fieldName);

  const _description = useMemo(
    () => description || fieldTemplate.description,
    [description, fieldTemplate.description]
  );

  return (
    <Flex id={id} className={NODE_FIELD_CLASS_NAME} flex="1 0 0">
      <FormControl flex="1 1 0" orientation="vertical">
        <NodeFieldElementLabel el={el} />
        <Flex w="full" gap={4}>
          <InputFieldRenderer
            nodeId={fieldIdentifier.nodeId}
            fieldName={fieldIdentifier.fieldName}
            settings={data.settings}
          />
        </Flex>
        {showDescription && _description && <FormHelperText>{_description}</FormHelperText>}
      </FormControl>
    </Flex>
  );
});
NodeFieldElementViewMode.displayName = 'NodeFieldElementViewMode';
