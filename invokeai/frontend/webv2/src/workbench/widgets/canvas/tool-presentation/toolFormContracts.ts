import type { ToolId } from '@workbench/canvas-engine/api';
import type { CanvasOperationState } from '@workbench/canvas-operations/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { ComponentType } from 'react';

export type CanvasToolOptionsEngine = Pick<
  CanvasEngineHandle,
  'document' | 'fonts' | 'interaction' | 'layers' | 'projectId' | 'selection' | 'tools' | 'viewport'
>;

export type CanvasOperationKind = Extract<CanvasOperationState, { status: 'active' }>['identity']['kind'];

export interface ToolFormProps {
  engine: CanvasToolOptionsEngine;
  /** Staging or generation owns the surface: mutating controls disable, Cancel stays. */
  isSurfaceInteractionLocked: boolean;
}

export interface ToolFooterProps {
  engine: CanvasToolOptionsEngine;
  isExternalInteractionLocked: boolean;
}

/** Preview cards take the footer's props: the engine plus the surface lock. */
export type ToolPreviewProps = ToolFooterProps;

export type ToolFormComponent = ComponentType<ToolFormProps>;

/**
 * One named group of a tool's property form. Group ids are GLOBAL keys: shared
 * across tools when (and only when) the tools share the group's component, so
 * React keeps the DOM alive across the tool switch and collapse state follows
 * the group rather than the tool.
 */
export interface ToolPropertyGroup {
  id: string;
  labelKey: string;
  /** Present makes the header a disclosure; the value is the default state. */
  collapsible?: 'open' | 'collapsed';
  body: ToolFormComponent;
}

/**
 * Describe tool panes with optional previews and typed row groups; the pane supports this form alongside migrating
 * region adapters.
 */
export interface ToolPropertyForm {
  id: ToolId;
  /** Paints into one leaf: a selected group gets the "select a layer" notice instead of strokes. */
  paintsLeaf?: boolean;
  preview?: ComponentType<ToolPreviewProps>;
  groups: readonly ToolPropertyGroup[];
  /** Sticks to the pane's bottom edge while the form scrolls: session status, Apply, Cancel. */
  footer?: ComponentType<ToolFooterProps>;
}

/** A guarded operation's pane form: named groups plus the sticky footer that owns its verbs. */
export interface OperationPropertyForm {
  kind: CanvasOperationKind;
  groups: readonly ToolPropertyGroup[];
  /** Status chip, Process/Reset, Apply and Cancel; pinned to the pane's bottom edge. */
  footer: ComponentType<ToolFooterProps>;
}
