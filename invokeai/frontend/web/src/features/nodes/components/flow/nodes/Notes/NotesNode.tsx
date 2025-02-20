import { Box, Flex, Textarea } from '@invoke-ai/ui-library';
import type { Node, NodeProps } from '@xyflow/react';
import { useAppDispatch } from 'app/store/storeHooks';
import NodeCollapseButton from 'features/nodes/components/flow/nodes/common/NodeCollapseButton';
import NodeTitle from 'features/nodes/components/flow/nodes/common/NodeTitle';
import NodeWrapper from 'features/nodes/components/flow/nodes/common/NodeWrapper';
import { notesNodeValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_PAN_CLASS } from 'features/nodes/types/constants';
import type { NotesNodeData } from 'features/nodes/types/invocation';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';

const NotesNode = (props: NodeProps<Node<NotesNodeData>>) => {
  const { id: nodeId, data, selected } = props;
  const { notes, isOpen } = data;
  const dispatch = useAppDispatch();
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(notesNodeValueChanged({ nodeId, value: e.target.value }));
    },
    [dispatch, nodeId]
  );

  return (
    <NodeWrapper nodeId={nodeId} selected={selected}>
      <Flex
        layerStyle="nodeHeader"
        borderTopRadius="base"
        borderBottomRadius={isOpen ? 0 : 'base'}
        alignItems="center"
        justifyContent="space-between"
        h={8}
      >
        <NodeCollapseButton nodeId={nodeId} isOpen={isOpen} />
        <NodeTitle nodeId={nodeId} title="Notes" />
        <Box minW={8} />
      </Flex>
      {isOpen && (
        <>
          <Flex
            layerStyle="nodeBody"
            className={NO_PAN_CLASS}
            cursor="auto"
            flexDirection="column"
            borderBottomRadius="base"
            w="full"
            h="full"
            p={2}
            gap={1}
          >
            <Flex className={NO_PAN_CLASS} w="full" h="full" flexDir="column">
              <Textarea
                className={NO_DRAG_CLASS}
                value={notes}
                onChange={handleChange}
                rows={8}
                resize="none"
                fontSize="sm"
              />
            </Flex>
          </Flex>
        </>
      )}
    </NodeWrapper>
  );
};

export default memo(NotesNode);
