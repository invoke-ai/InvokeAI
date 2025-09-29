import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasWorkflowModeProvider } from 'features/controlLayers/components/CanvasWorkflowModeContext';
import { CanvasWorkflowRootContainer } from 'features/controlLayers/components/CanvasWorkflowRootContainer';
import { selectCanvasWorkflowNodesSlice } from 'features/controlLayers/store/canvasWorkflowNodesSlice';
import { memo } from 'react';

/**
 * Renders the exposed fields for a canvas workflow.
 *
 * This component renders the workflow's form in view mode.
 * Each field element is wrapped with the appropriate InvocationNodeContext
 * in CanvasWorkflowFormElementComponent.
 */
export const CanvasWorkflowFieldsPanel = memo(() => {
  const nodesState = useAppSelector(selectCanvasWorkflowNodesSlice);

  // Check if form is empty
  const rootElement = nodesState.form.elements[nodesState.form.rootElementId];
  if (!rootElement || !('data' in rootElement) || !rootElement.data || !('children' in rootElement.data) || rootElement.data.children.length === 0) {
    return (
      <Flex w="full" p={4} justifyContent="center">
        <Text variant="subtext">No fields exposed in this workflow</Text>
      </Flex>
    );
  }

  return (
    <CanvasWorkflowModeProvider>
      <Flex w="full" justifyContent="center" p={4}>
        <CanvasWorkflowRootContainer />
      </Flex>
    </CanvasWorkflowModeProvider>
  );
});
CanvasWorkflowFieldsPanel.displayName = 'CanvasWorkflowFieldsPanel';