import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasWorkflowNodesSlice } from 'features/controlLayers/store/canvasWorkflowNodesSlice';
import {
  ContainerContextProvider,
  DepthContextProvider,
  useContainerContext,
  useDepthContext,
} from 'features/nodes/components/sidePanel/builder/contexts';
import { CONTAINER_CLASS_NAME, isContainerElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

import { CanvasWorkflowFormElementComponent } from './CanvasWorkflowFormElementComponent';

const containerViewModeSx: SystemStyleObject = {
  gap: 2,
  '&[data-self-layout="column"]': {
    flexDir: 'column',
    alignItems: 'stretch',
  },
  '&[data-self-layout="row"]': {
    flexDir: 'row',
    alignItems: 'flex-start',
    overflowX: 'auto',
    overflowY: 'visible',
    h: 'min-content',
    flexShrink: 0,
  },
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    flex: '1 1 0',
    minW: 32,
  },
};

/**
 * Container element for canvas workflow fields.
 * This reads from the canvas workflow nodes slice.
 */
export const CanvasWorkflowContainerElement = memo(({ id }: { id: string }) => {
  const nodesState = useAppSelector(selectCanvasWorkflowNodesSlice);
  const el = nodesState.form.elements[id];
  const depth = useDepthContext();
  const containerCtx = useContainerContext();

  if (!el || !isContainerElement(el)) {
    return null;
  }

  const { data } = el;
  const { children, layout } = data;

  return (
    <DepthContextProvider depth={depth + 1}>
      <ContainerContextProvider id={id} layout={layout}>
        <Flex
          id={id}
          className={CONTAINER_CLASS_NAME}
          sx={containerViewModeSx}
          data-self-layout={layout}
          data-depth={depth}
          data-parent-layout={containerCtx.layout}
        >
          {children.map((childId) => (
            <CanvasWorkflowFormElementComponent key={childId} id={childId} />
          ))}
        </Flex>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
CanvasWorkflowContainerElement.displayName = 'CanvasWorkflowContainerElement';
