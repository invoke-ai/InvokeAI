import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, useGlobalMenuClose } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useIsWorkflowEditorLocked } from 'features/nodes/hooks/useIsWorkflowEditorLocked';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { useNodeExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { useZoomToNode } from 'features/nodes/hooks/useZoomToNode';
import { selectNodeOpacity } from 'features/nodes/store/workflowSettingsSlice';
import { DRAG_HANDLE_CLASSNAME, NO_FIT_ON_DOUBLE_CLICK_CLASS, NODE_WIDTH } from 'features/nodes/types/constants';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { MouseEvent, PropsWithChildren } from 'react';
import { memo, useCallback } from 'react';

import { containerSx, inProgressSx, shadowsSx } from './shared';

type NonInvocationNodeWrapperProps = PropsWithChildren & {
  nodeId: string;
  selected: boolean;
  width?: ChakraProps['w'];
};

const NonInvocationNodeWrapper = (props: NonInvocationNodeWrapperProps) => {
  const { nodeId, width, children, selected } = props;
  const mouseOverNode = useMouseOverNode(nodeId);
  const zoomToNode = useZoomToNode(nodeId);
  const isLocked = useIsWorkflowEditorLocked();

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
      zoomToNode();
    },
    [zoomToNode]
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
      data-is-editor-locked={isLocked}
      data-is-selected={selected}
    >
      <Box sx={shadowsSx} />
      <Box sx={inProgressSx} data-is-in-progress={isInProgress} />
      {children}
      <Box className="node-selection-overlay" />
    </Box>
  );
};

export default memo(NonInvocationNodeWrapper);
