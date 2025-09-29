import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasWorkflowNodesSlice } from 'features/controlLayers/store/canvasWorkflowNodesSlice';
import { DividerElement } from 'features/nodes/components/sidePanel/builder/DividerElement';
import { HeadingElement } from 'features/nodes/components/sidePanel/builder/HeadingElement';
import { NodeFieldElementViewMode } from 'features/nodes/components/sidePanel/builder/NodeFieldElementViewMode';
import { TextElement } from 'features/nodes/components/sidePanel/builder/TextElement';
import {
  isContainerElement,
  isDividerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
} from 'features/nodes/types/workflow';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

import { CanvasWorkflowContainerElement } from './CanvasWorkflowContainerElement';
import { CanvasWorkflowInvocationNodeContextProvider } from './CanvasWorkflowInvocationContext';

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