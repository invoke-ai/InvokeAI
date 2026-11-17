import type { GraphPreviewInvoke } from '@features/workflow/contracts';
import type { InvocationSourceId } from '@workbench/invocationContracts';

import { type ComponentType, createContext, type ReactNode } from 'react';

export interface GraphPreviewInvokeScopeProps {
  sourceId: InvocationSourceId;
  /** Closed when a run starts, or an issue's action leads elsewhere. */
  onOpenChange: (isOpen: boolean) => void;
  /** Renders the preview with the Invoke made for it. */
  render: (invoke: GraphPreviewInvoke) => ReactNode;
}

/**
 * Where a widget's graph preview gets its Invoke: the shell provides the top bar's (`GraphPreviewInvokeScope`), so the
 * widget frame never imports the top bar's code. A frame importing it would share that boot code with the lazy preview
 * and split it out of the editor's boot chunk into requests of its own.
 */
export const GraphPreviewInvokeScopeContext = createContext<ComponentType<GraphPreviewInvokeScopeProps> | null>(null);
