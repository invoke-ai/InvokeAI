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
}

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
  preview?: ComponentType<ToolFormProps>;
  groups: readonly ToolPropertyGroup[];
  /** The form's last row, inside the Tool section: session status, Apply, Cancel. */
  actions?: ComponentType<ToolFormProps>;
}

/** What a running operation reports to its section, in the person's language. */
export interface OperationStatus {
  /** The layer it runs on (name, type, size): the section subtitle's tooltip. */
  source: string;
  /** The phase while work is in flight, beside the section title; null at rest. */
  busyText: string | null;
  /** The failure, shown after the groups; `detail` is the technical cause when it adds to `text`. */
  error: { text: string; detail: string | null } | null;
}

/** A guarded operation's pane form: named groups, its status, and the actions row that owns its verbs. */
export interface OperationPropertyForm {
  kind: CanvasOperationKind;
  groups: readonly ToolPropertyGroup[];
  /** A hook the Operation section calls on every render; null while the operation has no session yet. */
  useStatus(engine: CanvasToolOptionsEngine): OperationStatus | null;
  /** Reset, Cancel, the manual run, and Apply with its save-as menu; the section's last row before any error. */
  actions: ComponentType<ToolFormProps>;
}
