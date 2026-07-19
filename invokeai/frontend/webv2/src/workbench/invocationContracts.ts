export type InvocationSourceId = 'generate' | 'workflow' | 'upscale' | 'canvas';

export type InvocationMode = 'global' | 'dialog';

export type ResultDestination = 'canvas' | 'gallery';

export interface InvocationRoute {
  sourceId: InvocationSourceId;
  destination: ResultDestination;
  sourceLocked: boolean;
  destinationLocked: boolean;
}

export interface ResolvedInvocationRoute extends InvocationRoute {
  mode: InvocationMode;
  sourceValid: boolean;
  destinationValid: boolean;
  /** The top validation issue, shown on the fixed Invoke control's secondary line. */
  validationMessage?: string;
  /** Every reason the route cannot run right now (legacy `reasonsWhyCannotEnqueue` equivalent). */
  validationReasons: string[];
}

export interface InvocationControllerState extends InvocationRoute {
  lastSubmittedRunId?: string;
}
