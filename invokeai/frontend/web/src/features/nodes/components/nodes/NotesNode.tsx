import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAITextarea from 'common/components/IAITextarea';
import { notesNodeValueChanged } from 'features/nodes/store/nodesSlice';
import { NotesNodeData } from 'features/nodes/types/types';
import { ChangeEvent, memo, useCallback } from 'react';
import { NodeProps } from 'reactflow';
import NodeCollapseButton from '../Invocation/NodeCollapseButton';
import NodeTitle from '../Invocation/NodeTitle';
import NodeWrapper from '../Invocation/NodeWrapper';

const NotesNode = (props: NodeProps<NotesNodeData>) => {
  const { id: nodeId, data } = props;
  const { notes, isOpen } = data;
  const dispatch = useAppDispatch();
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(notesNodeValueChanged({ nodeId, value: e.target.value }));
    },
    [dispatch, nodeId]
  );

  return (
    <NodeWrapper nodeProps={props}>
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
        <NodeCollapseButton nodeProps={props} />
        <NodeTitle nodeData={props.data} title="Notes" />
        <Box minW={8} />
      </Flex>
      {isOpen && (
        <>
          <Flex
            layerStyle="nodeBody"
            className={'nopan'}
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
