import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useInputFieldTemplateTitleOrThrow } from 'features/nodes/hooks/useInputFieldTemplateTitleOrThrow';
import { useInputFieldUserTitleOrThrow } from 'features/nodes/hooks/useInputFieldUserTitleOrThrow';
import { useNodeTemplateTitleOrThrow } from 'features/nodes/hooks/useNodeTemplateTitleOrThrow';
import { useNodeUserTitleOrThrow } from 'features/nodes/hooks/useNodeUserTitleOrThrow';
import { selectNodeFieldElementsDeduped } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';

export const DeployWorkflowPanelContent = memo(() => {
  const nodeFieldElements = useAppSelector(selectNodeFieldElementsDeduped);
  return (
    <Flex flexDir="column">
      {nodeFieldElements.map((el) => {
        const { nodeId, fieldName } = el.data.fieldIdentifier;
        return <NodeFieldPreview key={`${nodeId}-${fieldName}`} nodeId={nodeId} fieldName={fieldName} />;
      })}
    </Flex>
  );
});
DeployWorkflowPanelContent.displayName = 'DeployWorkflowPanelContent';

const NodeFieldPreview = memo(({ nodeId, fieldName }: { nodeId: string; fieldName: string }) => {
  const nodeUserTitle = useNodeUserTitleOrThrow(nodeId);
  const nodeTemplateTitle = useNodeTemplateTitleOrThrow(nodeId);
  const fieldUserTitle = useInputFieldUserTitleOrThrow(nodeId, fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitleOrThrow(nodeId, fieldName);
  return (
    <Flex>
      <Text>{`${nodeUserTitle || nodeTemplateTitle} -> ${fieldUserTitle || fieldTemplateTitle}`}</Text>
      <Text>{`${nodeId} -> ${fieldName}`}</Text>
      <Text></Text>
    </Flex>
  );
});
NodeFieldPreview.displayName = 'NodeFieldPreview';
