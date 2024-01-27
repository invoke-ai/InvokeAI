import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, useGlobalMenuClose, useToken } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import NodeSelectionOverlay from 'common/components/NodeSelectionOverlay';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { nodeExclusivelySelected, selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { DRAG_HANDLE_CLASSNAME, NODE_WIDTH } from 'features/nodes/types/constants';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { MouseEvent, PropsWithChildren } from 'react';
import { memo, useCallback, useMemo } from 'react';

type NodeWrapperProps = PropsWithChildren & {
  nodeId: string;
  selected: boolean;
  width?: ChakraProps['w'];
};

const NodeWrapper = (props: NodeWrapperProps) => {
  const { nodeId, width, children, selected } = props;
  const { isMouseOverNode, handleMouseOut, handleMouseOver } = useMouseOverNode(nodeId);

  const selectIsInProgress = useMemo(
    () =>
      createSelector(
        selectNodesSlice,
        (nodes) => nodes.nodeExecutionStates[nodeId]?.status === zNodeStatus.enum.IN_PROGRESS
      ),
    [nodeId]
  );

  const isInProgress = useAppSelector(selectIsInProgress);

  const [nodeInProgress, shadowsXl, shadowsBase] = useToken('shadows', [
    'nodeInProgress',
    'shadows.xl',
    'shadows.base',
  ]);

  const dispatch = useAppDispatch();

  const opacity = useAppSelector((s) => s.nodes.nodeOpacity);
  const { onCloseGlobal } = useGlobalMenuClose();

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (!e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey) {
        dispatch(nodeExclusivelySelected(nodeId));
      }
      onCloseGlobal();
    },
    [dispatch, onCloseGlobal, nodeId]
  );

  return (
    <Box
      onClick={handleClick}
      onMouseEnter={handleMouseOver}
      onMouseLeave={handleMouseOut}
      className={DRAG_HANDLE_CLASSNAME}
      h="full"
      position="relative"
      borderRadius="base"
      w={width ? width : NODE_WIDTH}
      transitionProperty="common"
      transitionDuration="0.1s"
      cursor="grab"
      opacity={opacity}
    >
      <Box
        position="absolute"
        top={0}
        insetInlineEnd={0}
        bottom={0}
        insetInlineStart={0}
        borderRadius="base"
        pointerEvents="none"
        shadow={`${shadowsXl}, ${shadowsBase}, ${shadowsBase}`}
        zIndex={-1}
      />
      <Box
        position="absolute"
        top={0}
        insetInlineEnd={0}
        bottom={0}
        insetInlineStart={0}
        borderRadius="md"
        pointerEvents="none"
        transitionProperty="common"
        transitionDuration="0.1s"
        opacity={0.7}
        shadow={isInProgress ? nodeInProgress : undefined}
        zIndex={-1}
      />
      {children}
      <NodeSelectionOverlay isSelected={selected} isHovered={isMouseOverNode} />
    </Box>
  );
};

export default memo(NodeWrapper);
