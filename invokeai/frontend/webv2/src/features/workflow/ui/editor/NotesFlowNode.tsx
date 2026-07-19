import type { NodeProps } from '@xyflow/react';

import { Box, Input, Textarea } from '@chakra-ui/react';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { memo, useCallback, type ChangeEvent } from 'react';

import type { NotesFlowNode as NotesFlowNodeType } from './flowAdapters';

import { getWorkflowNodeChromeProps } from './nodeChrome';

const NotesFlowNodeComponent = ({ data, selected }: NodeProps<NotesFlowNodeType>) => {
  const { editGraph } = useProjectGraphCommands();
  const node = data.documentNode;
  const onLabelChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) =>
      editGraph({ label: event.currentTarget.value, nodeId: node.id, type: 'setNodeLabel' }),
    [editGraph, node.id]
  );
  const onNotesChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) =>
      editGraph({ nodeId: node.id, notes: event.currentTarget.value, type: 'setNodeNotes' }),
    [editGraph, node.id]
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
