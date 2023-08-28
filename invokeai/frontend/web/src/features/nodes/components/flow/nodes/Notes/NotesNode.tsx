import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAITextarea from 'common/components/IAITextarea';
import { notesNodeValueChanged } from 'features/nodes/store/nodesSlice';
import { NotesNodeData } from 'features/nodes/types/types';
import { ChangeEvent, memo, useCallback } from 'react';
import { NodeProps } from 'reactflow';
import NodeWrapper from '../common/NodeWrapper';
import NodeCollapseButton from '../common/NodeCollapseButton';
import NodeTitle from '../common/NodeTitle';

const NotesNode = (props: NodeProps<NotesNodeData>) => {
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
        sx={{
          borderTopRadius: 'base',
          borderBottomRadius: isOpen ? 0 : 'base',
          alignItems: 'center',
          justifyContent: 'space-between',
          h: 8,
        }}
      >
        <NodeCollapseButton nodeId={nodeId} isOpen={isOpen} />
        <NodeTitle nodeId={nodeId} title="Notes" />
        <Box minW={8} />
      </Flex>
      {isOpen && (
        <>
          <Flex
            layerStyle="nodeBody"
            className="nopan"
            sx={{
              cursor: 'auto',
              flexDirection: 'column',
              borderBottomRadius: 'base',
              w: 'full',
              h: 'full',
              p: 2,
              gap: 1,
            }}
          >
            <Flex
              className="nopan"
              sx={{ flexDir: 'column', w: 'full', h: 'full' }}
            >
              <IAITextarea
                value={notes}
                onChange={handleChange}
                rows={8}
                resize="none"
                sx={{ fontSize: 'xs' }}
              />
            </Flex>
          </Flex>
        </>
      )}
    </NodeWrapper>
  );
};

export default memo(NotesNode);
