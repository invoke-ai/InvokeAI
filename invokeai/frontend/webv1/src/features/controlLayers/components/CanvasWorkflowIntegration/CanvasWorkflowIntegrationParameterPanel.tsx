import { Box } from '@invoke-ai/ui-library';
import { WorkflowFormPreview } from 'features/controlLayers/components/CanvasWorkflowIntegration/WorkflowFormPreview';
import { memo } from 'react';

export const CanvasWorkflowIntegrationParameterPanel = memo(() => {
  return (
    <Box w="full">
      <WorkflowFormPreview />
    </Box>
  );
});

CanvasWorkflowIntegrationParameterPanel.displayName = 'CanvasWorkflowIntegrationParameterPanel';
