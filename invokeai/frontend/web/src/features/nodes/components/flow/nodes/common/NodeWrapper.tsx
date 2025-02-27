import type { ChakraProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, useGlobalMenuClose } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useMouseOverFormField, useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { useNodeExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { useZoomToNode } from 'features/nodes/hooks/useZoomToNode';
import { selectNodeOpacity } from 'features/nodes/store/workflowSettingsSlice';
import { DRAG_HANDLE_CLASSNAME, NO_FIT_ON_DOUBLE_CLICK_CLASS, NODE_WIDTH } from 'features/nodes/types/constants';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { MouseEvent, PropsWithChildren } from 'react';
import { memo, useCallback } from 'react';

type NodeWrapperProps = PropsWithChildren & {
  nodeId: string;
  selected: boolean;
  width?: ChakraProps['w'];
};

// Certain CSS transitions are disabled as a performance optimization - they can cause massive slowdowns in large
// workflows even when the animations are GPU-accelerated CSS.

const containerSx: SystemStyleObject = {
  h: 'full',
  position: 'relative',
  borderRadius: 'base',
  transitionProperty: 'none',
  cursor: 'grab',
  // The action buttons are hidden by default and shown on hover
  '& .node-selection-overlay': {
    display: 'none',
    position: 'absolute',
    top: 0,
    insetInlineEnd: 0,
    bottom: 0,
    insetInlineStart: 0,
    borderRadius: 'base',
    transitionProperty: 'none',
    pointerEvents: 'none',
    opacity: 0.5,
  },
  '&[data-is-mouse-over-node="true"] .node-selection-overlay': {
    opacity: 1,
    display: 'block',
  },
  '&[data-is-mouse-over-form-field="true"] .node-selection-overlay': {
    opacity: 1,
    display: 'block',
    bg: 'invokeBlueAlpha.100',
  },
  _hover: {
    '& .node-selection-overlay': {
      display: 'block',
      shadow: '0 0 0 2px var(--invoke-colors-blue-300)',
    },
    '&[data-is-selected="true"] .node-selection-overlay': {
      display: 'block',
      opacity: 1,
      shadow: '0 0 0 3px var(--invoke-colors-blue-300)',
    },
  },
  '&[data-is-selected="true"] .node-selection-overlay': {
    display: 'block',
    shadow: '0 0 0 3px var(--invoke-colors-blue-300)',
  },
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
  transitionProperty: 'none',
  opacity: 0.7,
  zIndex: -1,
  display: 'none',
  shadow: '0 0 0 2px var(--invoke-colors-yellow-400), 0 0 20px 2px var(--invoke-colors-orange-700)',
  '&[data-is-in-progress="true"]': {
    display: 'block',
  },
};

const NodeWrapper = (props: NodeWrapperProps) => {
  const { nodeId, width, children, selected } = props;
  const mouseOverNode = useMouseOverNode(nodeId);
  const mouseOverFormField = useMouseOverFormField(nodeId);
  const zoomToNode = useZoomToNode();

  const executionState = useNodeExecutionState(nodeId);
  const isInProgress = executionState?.status === zNodeStatus.enum.IN_PROGRESS;

  const opacity = useAppSelector(selectNodeOpacity);
  const globalMenu = useGlobalMenuClose();

  const onDoubleClick = useCallback(
    (e: MouseEvent) => {
      if (!(e.target instanceof HTMLElement)) {
        // We have to manually narrow the type here thanks to a TS quirk
        return;
      }
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement ||
        e.target instanceof HTMLButtonElement ||
        e.target instanceof HTMLAnchorElement
      ) {
        // Don't fit the view if the user is editing a text field, select, button, or link
        return;
      }
      if (e.target.closest(`.${NO_FIT_ON_DOUBLE_CLICK_CLASS}`) !== null) {
        // This target is marked as not fitting the view on double click
        return;
      }
      zoomToNode(nodeId);
    },
    [nodeId, zoomToNode]
  );

  return (
    <Box
      onClick={globalMenu.onCloseGlobal}
      onDoubleClick={onDoubleClick}
      onMouseOver={mouseOverNode.handleMouseOver}
      onMouseOut={mouseOverNode.handleMouseOut}
      className={DRAG_HANDLE_CLASSNAME}
      sx={containerSx}
      width={width || NODE_WIDTH}
      opacity={opacity}
      data-is-selected={selected}
      data-is-mouse-over-form-field={mouseOverFormField.isMouseOverFormField}
    >
      <Box sx={shadowsSx} />
      <Box sx={inProgressSx} data-is-in-progress={isInProgress} />
      {children}
      <Box className="node-selection-overlay" />
    </Box>
  );
};

export default memo(NodeWrapper);
