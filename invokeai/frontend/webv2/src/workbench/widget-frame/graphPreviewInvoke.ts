import type { GraphPreviewInvoke } from '@features/workflow/contracts';
import type { InvocationSourceId } from '@workbench/invocationContracts';

import { type ComponentType, createContext, type ReactNode } from 'react';

export interface GraphPreviewInvokeScopeProps {
  sourceId: InvocationSourceId;
  /** Closed when a run starts, or an issue's action leads elsewhere. */
  onOpenChange: (isOpen: boolean) => void;
  render: (invoke: GraphPreviewInvoke) => ReactNode;
}

/** The shell provides the top bar's Invoke here so the frame never imports top-bar code into boot. */
export const GraphPreviewInvokeScopeContext = createContext<ComponentType<GraphPreviewInvokeScopeProps> | null>(null);
