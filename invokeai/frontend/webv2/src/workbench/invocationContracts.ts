import type { ForLoopValidationReason } from '@features/workflow/utility';
import type { ControlLayerIssue } from '@workbench/controlLayerChecks';

export type InvocationSourceId = 'generate' | 'workflow' | 'upscale' | 'video' | 'canvas';

export type InvocationMode = 'global' | 'dialog';

export type ResultDestination = 'canvas' | 'gallery';

export interface InvocationRoute {
  sourceId: InvocationSourceId;
  destination: ResultDestination;
  sourceLocked: boolean;
  destinationLocked: boolean;
}

/** A blocking control layer, kept structured so the shell words it from the locale. */
export interface ControlLayerValidationReason {
  controlLayerIssue: ControlLayerIssue;
}

export type InvocationValidationReason = string | ForLoopValidationReason | ControlLayerValidationReason;

export interface ResolvedInvocationRoute extends InvocationRoute {
  mode: InvocationMode;
  sourceValid: boolean;
  destinationValid: boolean;
  /** The top validation issue, shown on the fixed Invoke control's secondary line. */
  validationMessage?: InvocationValidationReason;
  /** Every reason the route cannot run right now (legacy `reasonsWhyCannotEnqueue` equivalent). */
  validationReasons: InvocationValidationReason[];
  /** Present when the workflow has batch nodes; size is null until an async generator resolves on invoke. */
  workflowBatch?: { size: number | null };
}

export interface InvocationControllerState extends InvocationRoute {
  lastSubmittedRunId?: string;
}
