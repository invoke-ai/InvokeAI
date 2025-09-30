import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasWorkflowNodesSlice } from 'features/controlLayers/store/canvasWorkflowNodesSlice';
import { ContainerContextProvider, DepthContextProvider } from 'features/nodes/components/sidePanel/builder/contexts';
import { isContainerElement, ROOT_CONTAINER_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo } from 'react';

import { CanvasWorkflowFormElementComponent } from './CanvasWorkflowFormElementComponent';

const rootViewModeSx: SystemStyleObject = {
  position: 'relative',
  alignItems: 'center',
  borderRadius: 'base',
  w: 'full',
  h: 'full',
  gap: 2,
  display: 'flex',
  flex: 1,
  maxW: '768px',
  '&[data-self-layout="column"]': {
    flexDir: 'column',
    alignItems: 'stretch',
  },
  '&[data-self-layout="row"]': {
    flexDir: 'row',
    alignItems: 'flex-start',
  },
};

/**
 * Root container for canvas workflow fields.
 * This reads from the canvas workflow nodes slice instead of the main nodes slice.
 */
export const CanvasWorkflowRootContainer = memo(() => {
  const nodesState = useAppSelector(selectCanvasWorkflowNodesSlice);
  const el = nodesState.form.elements[nodesState.form.rootElementId];

  if (!el || !isContainerElement(el)) {
    return null;
  }

  const { id, data } = el;
  const { children, layout } = data;

  return (
    <DepthContextProvider depth={0}>
      <ContainerContextProvider id={id} layout={layout}>
        <Box id={id} className={ROOT_CONTAINER_CLASS_NAME} sx={rootViewModeSx} data-self-layout={layout} data-depth={0}>
          {children.map((childId) => (
            <CanvasWorkflowFormElementComponent key={childId} id={childId} />
          ))}
        </Box>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
CanvasWorkflowRootContainer.displayName = 'CanvasWorkflowRootContainer';
