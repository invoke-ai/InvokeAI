import { OutputFieldHandle } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldHandle';
import { OutputFieldTitle } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldTitle';
import { OutputFieldWrapper } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldWrapper';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
}>;

export const OutputFieldNodesEditorView = memo(({ nodeId, fieldName }: Props) => {
  return (
    <OutputFieldWrapper>
      <OutputFieldTitle nodeId={nodeId} fieldName={fieldName} />
      <OutputFieldHandle nodeId={nodeId} fieldName={fieldName} />
    </OutputFieldWrapper>
  );
});

OutputFieldNodesEditorView.displayName = 'OutputFieldNodesEditorView';
