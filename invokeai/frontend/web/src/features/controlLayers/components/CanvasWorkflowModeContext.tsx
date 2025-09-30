import { CanvasWorkflowModeContext } from 'features/nodes/hooks/useWorkflowMode';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

/**
 * Context provider to override the workflow mode for canvas workflows.
 * Canvas workflows should always render fields in view mode, regardless of
 * the workflow tab's current mode.
 */

export const CanvasWorkflowModeProvider = memo(({ children }: PropsWithChildren) => {
  return <CanvasWorkflowModeContext.Provider value="view">{children}</CanvasWorkflowModeContext.Provider>;
});
CanvasWorkflowModeProvider.displayName = 'CanvasWorkflowModeProvider';
