import type { ChakraProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, useGlobalMenuClose } from '@invoke-ai/ui-library';
import type { NodeChange } from '@xyflow/react';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { useNodeExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { nodesChanged } from 'features/nodes/store/nodesSlice';
import { selectNodes } from 'features/nodes/store/selectors';
import { selectNodeOpacity } from 'features/nodes/store/workflowSettingsSlice';
import { DRAG_HANDLE_CLASSNAME, NODE_WIDTH } from 'features/nodes/types/constants';
import type { AnyNode } from 'features/nodes/types/invocation';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { MouseEvent, PropsWithChildren } from 'react';
import { memo, useCallback } from 'react';

type NodeWrapperProps = PropsWithChildren & {
  nodeId: string;
  selected: boolean;
  width?: ChakraProps['w'];
};

const containerSx: SystemStyleObject = {
  h: 'full',
  position: 'relative',
  borderRadius: 'base',
  transitionProperty: 'common',
  transitionDuration: '0.1s',
  cursor: 'grab',
};

const shadowsSx: SystemStyleObject = {
  position: 'absolute',
  top: 0,
  insetInlineEnd: 0,
  bottom: 0,
  insetInlineStart: 0,
  borderRadius: 'base',
  pointerEvents: 'none',
  zIndex: -1,
  shadow: 'var(--invoke-shadows-xl), var(--invoke-shadows-base), var(--invoke-shadows-base)',
};

const inProgressSx: SystemStyleObject = {
  position: 'absolute',
  top: 0,
  insetInlineEnd: 0,
  bottom: 0,
  insetInlineStart: 0,
  borderRadius: 'md',
  pointerEvents: 'none',
  transitionProperty: 'common',
  transitionDuration: '0.1s',
  opacity: 0.7,
  zIndex: -1,
  visibility: 'hidden',
  shadow: '0 0 0 2px var(--invoke-colors-yellow-400), 0 0 20px 2px var(--invoke-colors-orange-700)',
  '&[data-is-in-progress="true"]': {
    visibility: 'visible',
  },
};

const selectionOverlaySx: SystemStyleObject = {
  position: 'absolute',
  top: 0,
  insetInlineEnd: 0,
  bottom: 0,
  insetInlineStart: 0,
  borderRadius: 'base',
  transitionProperty: 'common',
  transitionDuration: '0.1s',
  pointerEvents: 'none',
  visibility: 'hidden',
  opacity: 0.5,
  '&[data-is-selected="true"], &[data-is-hovered="true"]': { visibility: 'visible' },
  '&[data-is-selected="true"]': { shadow: '0 0 0 3px var(--invoke-colors-blue-300)' },
  '&[data-is-hovered="true"]': { shadow: '0 0 0 2px var(--invoke-colors-blue-300)' },
  '&[data-is-selected="true"][data-is-hovered="true"]': {
    opacity: 1,
    shadow: '0 0 0 3px var(--invoke-colors-blue-300)',
  },
};

const NodeWrapper = (props: NodeWrapperProps) => {
  const { nodeId, width, children, selected } = props;
  const store = useAppStore();
  const { isMouseOverNode, handleMouseOut, handleMouseOver } = useMouseOverNode(nodeId);

  const executionState = useNodeExecutionState(nodeId);
  const isInProgress = executionState?.status === zNodeStatus.enum.IN_PROGRESS;

  const dispatch = useAppDispatch();

  const opacity = useAppSelector(selectNodeOpacity);
  const { onCloseGlobal } = useGlobalMenuClose();

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (!e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey) {
        const nodes = selectNodes(store.getState());
        const nodeChanges: NodeChange<AnyNode>[] = [];
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
      sx={containerSx}
      width={width || NODE_WIDTH}
      opacity={opacity}
    >
      <Box sx={shadowsSx} />
      <Box sx={inProgressSx} data-is-in-progress={isInProgress} />
      {children}
      <Box sx={selectionOverlaySx} data-is-selected={selected} data-is-hovered={isMouseOverNode} />
    </Box>
  );
};

export default memo(NodeWrapper);
