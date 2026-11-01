import type { GenerationIssue } from '@features/generation/contracts';
import type { WorkflowReadinessReason } from '@features/workflow/contracts';
import type { ControlLayerIssue } from '@workbench/controlLayerChecks';

export type InvocationSourceId = 'generate' | 'workflow' | 'upscale' | 'video' | 'canvas';

export type InvocationMode = 'global' | 'dialog';

export type ResultDestination = 'canvas' | 'gallery';

/** A dynamic prompt expansion, with the sample seed it drew from (null for All combinations). */
export interface ExpandedPositivePrompts {
  positivePrompts: string[];
  /** Recall reproduces this draw, and a resampling submission rotates only while the project still holds it. */
  positivePromptsSampleSeed: number | null;
}

/** The metadata a Model only upscale copies, with the image it was read from, so a changed input is never mislabelled. */
export interface UpscaleSourceMetadata {
  imageName: string;
  /** Null when the image has none. */
  metadata: Record<string, unknown> | null;
}

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

/**
 * A Generate settings issue, kept structured: the shell words it from the locale today, and its `target` (section and
 * field) and `fix` are there for a surface that jumps to the field or applies the fix.
 */
export interface GenerationValidationReason {
  generationIssue: GenerationIssue;
}

/** A problem with the route itself rather than its source's settings, kept structured so Invoke can offer the fix. */
export type RouteIssue =
  /** The source's widget is in no region or window of this project. */
  | { kind: 'sourceNotMounted'; sourceId: InvocationSourceId }
  /** Canvas staging takes images only. */
  | { kind: 'videoNeedsGallery' }
  | { kind: 'emptyCanvasFrame' };

export interface RouteValidationReason {
  routeIssue: RouteIssue;
}

export type InvocationValidationReason =
  | string
  | WorkflowReadinessReason
  | ControlLayerValidationReason
  | GenerationValidationReason
  | RouteValidationReason;

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
