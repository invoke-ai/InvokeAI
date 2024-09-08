import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, useGlobalMenuClose, useToken } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import NodeSelectionOverlay from 'common/components/NodeSelectionOverlay';
import { useExecutionState } from 'features/nodes/hooks/useExecutionState';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { nodesChanged } from 'features/nodes/store/nodesSlice';
import { selectNodes } from 'features/nodes/store/selectors';
import { selectNodeOpacity } from 'features/nodes/store/workflowSettingsSlice';
import { DRAG_HANDLE_CLASSNAME, NODE_WIDTH } from 'features/nodes/types/constants';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { MouseEvent, PropsWithChildren } from 'react';
import { memo, useCallback } from 'react';
import type { NodeChange } from 'reactflow';

type NodeWrapperProps = PropsWithChildren & {
  nodeId: string;
  selected: boolean;
  width?: ChakraProps['w'];
};

const NodeWrapper = (props: NodeWrapperProps) => {
  const { nodeId, width, children, selected } = props;
  const store = useAppStore();
  const { isMouseOverNode, handleMouseOut, handleMouseOver } = useMouseOverNode(nodeId);

  const executionState = useExecutionState(nodeId);
  const isInProgress = executionState?.status === zNodeStatus.enum.IN_PROGRESS;

  const [nodeInProgress, shadowsXl, shadowsBase] = useToken('shadows', [
    'nodeInProgress',
    'shadows.xl',
    'shadows.base',
  ]);

  const dispatch = useAppDispatch();

  const opacity = useAppSelector(selectNodeOpacity);
  const { onCloseGlobal } = useGlobalMenuClose();

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (!e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey) {
        const nodes = selectNodes(store.getState());
        const nodeChanges: NodeChange[] = [];
        nodes.forEach(({ id, selected }) => {
          if (selected !== (id === nodeId)) {
            nodeChanges.push({ type: 'select', id, selected: id === nodeId });
          }
        });
        if (nodeChanges.length > 0) {
          dispatch(nodesChanged(nodeChanges));
        }
      }
      onCloseGlobal();
    },
    [onCloseGlobal, store, dispatch, nodeId]
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
