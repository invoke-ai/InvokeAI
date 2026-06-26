import type { NodeProps } from '@xyflow/react';

import { Box, Input, Textarea } from '@chakra-ui/react';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { memo, useCallback, type ChangeEvent } from 'react';

import type { NotesFlowNode as NotesFlowNodeType } from './flowAdapters';

import { getWorkflowNodeChromeProps } from './nodeChrome';

const NotesFlowNodeComponent = ({ data, selected }: NodeProps<NotesFlowNodeType>) => {
  const dispatch = useWorkbenchDispatch();
  const node = data.documentNode;
  const onLabelChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) =>
      dispatch({
        action: { label: event.currentTarget.value, nodeId: node.id, type: 'setNodeLabel' },
        type: 'applyProjectGraphAction',
      }),
    [dispatch, node.id]
  );
  const onNotesChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) =>
      dispatch({
        action: { nodeId: node.id, notes: event.currentTarget.value, type: 'setNodeNotes' },
        type: 'applyProjectGraphAction',
      }),
    [dispatch, node.id]
  );

  return (
    <Box bg="bg.subtle" p="2" rounded="lg" w="16rem" {...getWorkflowNodeChromeProps({ selected })}>
      <Input
        aria-label="Note title"
        className="nodrag"
        fontWeight="700"
        mb="1.5"
        size="2xs"
        value={node.data.label}
        variant="flushed"
        onChange={onLabelChange}
      />
      <Textarea
        aria-label="Note text"
        className="nodrag nowheel"
        fontSize="2xs"
        minH="5rem"
        placeholder="Write a note…"
        resize="vertical"
        size="xs"
        value={node.data.notes}
        onChange={onNotesChange}
      />
    </Box>
  );
};

export const NotesFlowNode = memo(NotesFlowNodeComponent);
