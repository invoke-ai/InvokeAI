import {
  Box,
  ChakraProps,
  useColorModeValue,
  useToken,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import NodeSelectionOverlay from 'common/components/NodeSelectionOverlay';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { nodeExclusivelySelected } from 'features/nodes/store/nodesSlice';
import {
  DRAG_HANDLE_CLASSNAME,
  NODE_WIDTH,
} from 'features/nodes/types/constants';
import { NodeStatus } from 'features/nodes/types/types';
import { contextMenusClosed } from 'features/ui/store/uiSlice';
import {
  MouseEvent,
  PropsWithChildren,
  memo,
  useCallback,
  useMemo,
} from 'react';

type NodeWrapperProps = PropsWithChildren & {
  nodeId: string;
  selected: boolean;
  width?: NonNullable<ChakraProps['sx']>['w'];
};

const NodeWrapper = (props: NodeWrapperProps) => {
  const { nodeId, width, children, selected } = props;
  const { isMouseOverNode, handleMouseOut, handleMouseOver } =
    useMouseOverNode(nodeId);

  const selectIsInProgress = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) =>
          nodes.nodeExecutionStates[nodeId]?.status === NodeStatus.IN_PROGRESS
      ),
    [nodeId]
  );

  const isInProgress = useAppSelector(selectIsInProgress);

  const [nodeInProgressLight, nodeInProgressDark, shadowsXl, shadowsBase] =
    useToken('shadows', [
      'nodeInProgress.light',
      'nodeInProgress.dark',
      'shadows.xl',
      'shadows.base',
    ]);

  const dispatch = useAppDispatch();

  const inProgressShadow = useColorModeValue(
    nodeInProgressLight,
    nodeInProgressDark
  );

  const opacity = useAppSelector((state) => state.nodes.nodeOpacity);

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (!e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey) {
        dispatch(nodeExclusivelySelected(nodeId));
      }
      dispatch(contextMenusClosed());
    },
    [dispatch, nodeId]
  );

  return (
    <Box
      onClick={handleClick}
      onMouseEnter={handleMouseOver}
      onMouseLeave={handleMouseOut}
      className={DRAG_HANDLE_CLASSNAME}
      sx={{
        h: 'full',
        position: 'relative',
        borderRadius: 'base',
        w: width ?? NODE_WIDTH,
        transitionProperty: 'common',
        transitionDuration: '0.1s',
        cursor: 'grab',
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
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineEnd: 0,
          bottom: 0,
          insetInlineStart: 0,
          borderRadius: 'md',
          pointerEvents: 'none',
          transitionProperty: 'common',
          transitionDuration: 'normal',
          opacity: 0.7,
          shadow: isInProgress ? inProgressShadow : undefined,
          zIndex: -1,
        }}
      />
      {children}
      <NodeSelectionOverlay isSelected={selected} isHovered={isMouseOverNode} />
    </Box>
  );
};

export default memo(NodeWrapper);
