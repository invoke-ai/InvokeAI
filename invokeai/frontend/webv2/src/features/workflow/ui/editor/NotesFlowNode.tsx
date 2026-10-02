import type { NodeProps } from '@xyflow/react';

import { Box, Input, Text, Textarea } from '@chakra-ui/react';
import { getWorkflowNodeChromeProps, useIsWorkflowImageExport } from '@features/workflow/ui/nodeChrome';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { memo, useCallback, type ChangeEvent } from 'react';

import type { NotesFlowNode as NotesFlowNodeType } from './flowAdapters';

const NotesSnapshotNode = ({ data }: NodeProps<NotesFlowNodeType>) => {
  const node = data.documentNode;

  return (
    <Box
      bg="bg.subtle"
      data-workflow-export-static-node-content="true"
      data-workflow-node-shell="true"
      p="2"
      rounded="lg"
      w="16rem"
      {...getWorkflowNodeChromeProps({ selected: false })}
    >
      <MiddleTruncate
        data-workflow-export-node-title="true"
        fontSize="2xs"
        fontWeight="700"
        mb="1.5"
        text={node.data.label}
      />
      {node.data.notes ? (
        <Text fontSize="2xs" overflowWrap="anywhere" whiteSpace="pre-wrap">
          {node.data.notes}
        </Text>
      ) : null}
    </Box>
  );
};

const NotesEditorNode = ({ data, selected }: NodeProps<NotesFlowNodeType>) => {
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
    <Box
      bg="bg.subtle"
      data-is-selected={selected}
      data-workflow-node-shell="true"
      p="2"
      rounded="lg"
      w="16rem"
      {...getWorkflowNodeChromeProps({ selected })}
    >
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

const NotesFlowNodeComponent = (props: NodeProps<NotesFlowNodeType>) => {
  const isWorkflowImageExport = useIsWorkflowImageExport();

  return isWorkflowImageExport ? <NotesSnapshotNode {...props} /> : <NotesEditorNode {...props} />;
};

export const NotesFlowNode = memo(NotesFlowNodeComponent);
