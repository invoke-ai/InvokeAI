import type { NodeProps } from '@xyflow/react';

import { Box, Input, Textarea } from '@chakra-ui/react';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { memo, type ChangeEvent } from 'react';

import type { NotesFlowNode as NotesFlowNodeType } from './flowAdapters';

const NotesFlowNodeComponent = ({ data, selected }: NodeProps<NotesFlowNodeType>) => {
  const dispatch = useWorkbenchDispatch();
  const node = data.documentNode;

  return (
    <Box
      bg="bg.subtle"
      borderColor={selected ? 'accent.solid' : 'border.emphasized'}
      borderWidth="1px"
      p="2"
      rounded="lg"
      shadow={selected ? 'md' : 'sm'}
      w="16rem"
    >
      <Input
        aria-label="Note title"
        className="nodrag"
        fontWeight="700"
        mb="1.5"
        size="2xs"
        value={node.data.label}
        variant="flushed"
        onChange={(event: ChangeEvent<HTMLInputElement>) =>
          dispatch({
            action: { label: event.currentTarget.value, nodeId: node.id, type: 'setNodeLabel' },
            type: 'applyProjectGraphAction',
          })
        }
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
        onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
          dispatch({
            action: { nodeId: node.id, notes: event.currentTarget.value, type: 'setNodeNotes' },
            type: 'applyProjectGraphAction',
          })
        }
      />
    </Box>
  );
};

export const NotesFlowNode = memo(NotesFlowNodeComponent);
