import {
  Box,
  ChakraProps,
  useColorModeValue,
  useToken,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { nodeClicked } from 'features/nodes/store/nodesSlice';
import { MouseEvent, PropsWithChildren, useCallback, useMemo } from 'react';
import { DRAG_HANDLE_CLASSNAME, NODE_WIDTH } from '../../types/constants';
import { NodeData } from 'features/nodes/types/types';
import { NodeProps } from 'reactflow';

const useNodeSelect = (nodeId: string) => {
  const dispatch = useAppDispatch();

  const selectNode = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      dispatch(nodeClicked({ nodeId, ctrlOrMeta: e.ctrlKey || e.metaKey }));
    },
    [dispatch, nodeId]
  );

  return selectNode;
};

type NodeWrapperProps = PropsWithChildren & {
  nodeProps: NodeProps<NodeData>;
  width?: NonNullable<ChakraProps['sx']>['w'];
};

const NodeWrapper = (props: NodeWrapperProps) => {
  const { width, children, nodeProps } = props;
  const { data, selected } = nodeProps;
  const nodeId = data.id;

  const [
    nodeSelectedOutlineLight,
    nodeSelectedOutlineDark,
    shadowsXl,
    shadowsBase,
  ] = useToken('shadows', [
    'nodeSelectedOutline.light',
    'nodeSelectedOutline.dark',
    'shadows.xl',
    'shadows.base',
  ]);

  const selectNode = useNodeSelect(nodeId);

  const shadow = useColorModeValue(
    nodeSelectedOutlineLight,
    nodeSelectedOutlineDark
  );

  const shift = useAppSelector((state) => state.hotkeys.shift);
  const opacity = useAppSelector((state) => state.nodes.nodeOpacity);
  const className = useMemo(
    () => (shift ? DRAG_HANDLE_CLASSNAME : 'nopan'),
    [shift]
  );

  return (
    <Box
      onClickCapture={selectNode}
      className={className}
      sx={{
        h: 'full',
        position: 'relative',
        borderRadius: 'base',
        w: width ?? NODE_WIDTH,
        transitionProperty: 'common',
        transitionDuration: '0.1s',
        shadow: selected ? shadow : undefined,
        opacity,
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineEnd: 0,
          bottom: 0,
          insetInlineStart: 0,
          borderRadius: 'base',
          pointerEvents: 'none',
          shadow: `${shadowsXl}, ${shadowsBase}, ${shadowsBase}`,
          zIndex: -1,
        }}
      />
      {children}
    </Box>
  );
};

export default NodeWrapper;
