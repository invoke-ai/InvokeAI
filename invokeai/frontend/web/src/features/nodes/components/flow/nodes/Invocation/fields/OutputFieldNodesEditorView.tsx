import { OutputFieldHandle } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldHandle';
import { OutputFieldTitle } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldTitle';
import { OutputFieldWrapper } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldWrapper';
import { useOutputFieldConnectionState } from 'features/nodes/hooks/useOutputFieldConnectionState';
import { useOutputFieldIsConnected } from 'features/nodes/hooks/useOutputFieldIsConnected';
import { useOutputFieldTemplate } from 'features/nodes/hooks/useOutputFieldTemplate';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
}>;

export const OutputFieldNodesEditorView = memo(({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useOutputFieldTemplate(nodeId, fieldName);
  const isConnected = useOutputFieldIsConnected(nodeId, fieldName);
  const { isConnectionInProgress, isConnectionStartField, validationResult } = useOutputFieldConnectionState(
    nodeId,
    fieldName
  );

  return (
    <OutputFieldWrapper>
      <OutputFieldTitle
        nodeId={nodeId}
        fieldName={fieldName}
        isDisabled={(isConnectionInProgress && !validationResult.isValid && !isConnectionStartField) || isConnected}
      />
      <OutputFieldHandle
        fieldTemplate={fieldTemplate}
        isConnectionInProgress={isConnectionInProgress}
        isConnectionStartField={isConnectionStartField}
        validationResult={validationResult}
      />
    </OutputFieldWrapper>
  );
});

OutputFieldNodesEditorView.displayName = 'OutputFieldNodesEditorView';
