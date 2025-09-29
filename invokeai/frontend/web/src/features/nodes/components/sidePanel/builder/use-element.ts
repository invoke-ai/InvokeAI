import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasWorkflowElement } from 'features/controlLayers/components/CanvasWorkflowElementContext';
import { buildSelectElement } from 'features/nodes/store/selectors';
import type { FormElement } from 'features/nodes/types/workflow';
import { useMemo } from 'react';

export const useElement = (id: string): FormElement | undefined => {
  const canvasGetElement = useCanvasWorkflowElement();
  const selector = useMemo(() => buildSelectElement(id), [id]);
  const regularElement = useAppSelector(selector);

  // If we're in canvas workflow context, use that; otherwise use regular nodes
  if (canvasGetElement) {
    return canvasGetElement(id);
  }

  return regularElement;
};
