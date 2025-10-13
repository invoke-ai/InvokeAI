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
import { DividerElement } from 'features/nodes/components/sidePanel/builder/DividerElement';
import { HeadingElement } from 'features/nodes/components/sidePanel/builder/HeadingElement';
import { NodeFieldElementViewMode } from 'features/nodes/components/sidePanel/builder/NodeFieldElementViewMode';
import { TextElement } from 'features/nodes/components/sidePanel/builder/TextElement';
import {
  CONTAINER_CLASS_NAME,
  isContainerElement,
  isDividerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
} from 'features/nodes/types/workflow';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

import { CanvasWorkflowInvocationNodeContextProvider } from './CanvasWorkflowInvocationContext';

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
const CanvasWorkflowContainerElement = memo(({ id }: { id: string }) => {
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

/**
 * Renders a form element from canvas workflow nodes.
 * Recursively handles all element types.
 */
export const CanvasWorkflowFormElementComponent = memo(({ id }: { id: string }) => {
  const nodesState = useAppSelector(selectCanvasWorkflowNodesSlice);
  const el = nodesState.form.elements[id];

  if (!el) {
    return null;
  }

  if (isContainerElement(el)) {
    return <CanvasWorkflowContainerElement key={id} id={id} />;
  }

  if (isNodeFieldElement(el)) {
    return (
      <CanvasWorkflowInvocationNodeContextProvider key={id} nodeId={el.data.fieldIdentifier.nodeId}>
        <NodeFieldElementViewMode el={el} />
      </CanvasWorkflowInvocationNodeContextProvider>
    );
  }

  if (isDividerElement(el)) {
    return <DividerElement key={id} id={id} />;
  }

  if (isHeadingElement(el)) {
    return <HeadingElement key={id} id={id} />;
  }

  if (isTextElement(el)) {
    return <TextElement key={id} id={id} />;
  }

  assert<Equals<typeof el, never>>(false, `Unhandled type for element with id ${id}`);
});
CanvasWorkflowFormElementComponent.displayName = 'CanvasWorkflowFormElementComponent';
