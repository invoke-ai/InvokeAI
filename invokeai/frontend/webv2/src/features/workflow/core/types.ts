/**
 * Editable legacy-compatible workflow documents compile only at invocation time so queued GraphContract snapshots
 * remain immutable.
 */

import type { SeedMode } from '@platform/core/seed';

export type FieldCardinality = 'SINGLE' | 'COLLECTION' | 'SINGLE_OR_COLLECTION';

export interface FieldType {
  name: string;
  cardinality: FieldCardinality;
  batch: boolean;
  /** Set when a `ui_type` override replaced the parsed type; both sides count for connection equality. */
  originalType?: FieldType;
}

export interface FieldInputTemplate {
  name: string;
  title: string;
  description: string;
  type: FieldType;
  required: boolean;
  /** Internal inputs are backend-provided but connectable; skip them when guessing an authored primary input. */
  fieldKind: 'input' | 'internal';
  /** How the field receives data: only via edge, only direct value, or either. */
  input: 'connection' | 'direct' | 'any';
  default?: unknown;
  uiHidden: boolean;
  uiOrder: number | null;
  uiComponent: 'slider' | 'textarea' | 'video-frame-index' | null;
  uiChoiceLabels: Record<string, string> | null;
  /** Enum choices when the field is an EnumField. Values retain backend types. */
  options: unknown[] | null;
  minimum: number | null;
  maximum: number | null;
  exclusiveMinimum: number | null;
  exclusiveMaximum: number | null;
  multipleOf: number | null;
  /** Item-count bounds of a collection; absent on templates built before lists were editable. */
  minItems?: number | null;
  maxItems?: number | null;
  /** String length bounds, applied per item for string collections. */
  minLength?: number | null;
  maxLength?: number | null;
  uiModelBase: string[] | null;
  uiModelFormat: string[] | null;
  uiModelType: string[] | null;
}

export interface FieldOutputTemplate {
  name: string;
  title: string;
  description: string;
  type: FieldType;
  outputScope?: 'iteration' | 'final';
  uiHidden?: boolean;
}

export interface InvocationTemplate {
  type: string;
  title: string;
  description: string;
  tags: string[];
  category: string;
  version: string;
  useCache: boolean;
  nodePack: string;
  classification: string;
  inputs: Record<string, FieldInputTemplate>;
  outputs: Record<string, FieldOutputTemplate>;
  outputType: string;
}

export type InvocationTemplates = Record<string, InvocationTemplate>;

export interface XYPosition {
  x: number;
  y: number;
}

/** A direct input value on a node. Connection-only fields keep `value` undefined. */
export interface WorkflowFieldInstance {
  name: string;
  label: string;
  /** True when the label was explicitly changed by the user rather than generated from a template. */
  labelOverride?: boolean;
  /** User override of the template's field description (shown in the Linear UI). */
  description?: string;
  /** True when the description was explicitly changed by the user, including clearing it. */
  descriptionOverride?: boolean;
  /** Absent seedMode means fixed, including older documents and legacy readers that strip it. */
  seedMode?: SeedMode;
  value?: unknown;
}

/** One seed input's move after a submission; applied only while the field still holds `fromSeed` under `seedMode`. */
export interface WorkflowSeedFieldAdvance {
  fieldName: string;
  /** The value the field held when planned; absent when it was empty and the sequence started from the template default. */
  fromSeed?: number;
  nodeId: string;
  seedMode: SeedMode;
  toSeed: number;
}

export interface WorkflowInvocationNodeData {
  type: string;
  version: string;
  label: string;
  notes: string;
  isOpen: boolean;
  isIntermediate: boolean;
  useCache: boolean;
  nodePack: string;
  inputs: Record<string, WorkflowFieldInstance>;
  /** Persisted templates for fields exposed by the selected saved workflow. */
  dynamicInputTemplates?: Record<string, FieldInputTemplate>;
  /** Runtime reconciliation state for the selected saved workflow. */
  callSavedWorkflowStatus?: 'loading' | 'ready' | 'error';
}

export interface WorkflowInvocationNode {
  id: string;
  type: 'invocation';
  position: XYPosition;
  data: WorkflowInvocationNodeData;
}

export interface WorkflowNotesNode {
  id: string;
  type: 'notes';
  position: XYPosition;
  data: {
    label: string;
    notes: string;
  };
}

/** UI-only node mirroring the legacy `current_image` node: shows the latest run output / progress image. */
export interface WorkflowCurrentImageNode {
  id: string;
  type: 'current_image';
  position: XYPosition;
  data: {
    label: string;
  };
}

export interface WorkflowConnectorNode {
  id: string;
  type: 'connector';
  position: XYPosition;
  data: {
    label: string;
  };
}

export type WorkflowNode =
  | WorkflowInvocationNode
  | WorkflowNotesNode
  | WorkflowCurrentImageNode
  | WorkflowConnectorNode;

export interface WorkflowEdge {
  id: string;
  type: 'default' | 'loop_linkage';
  source: string;
  target: string;
  sourceHandle: string;
  targetHandle: string;
}

export interface FieldIdentifier {
  nodeId: string;
  fieldName: string;
}

export interface ContainerFormElement {
  id: string;
  type: 'container';
  parentId?: string;
  data: {
    layout: 'row' | 'column';
    children: string[];
  };
}

export interface NodeFieldFormElement {
  id: string;
  type: 'node-field';
  parentId?: string;
  data: {
    fieldIdentifier: FieldIdentifier;
    showDescription: boolean;
    /** Shows a randomize button beside numeric fields in the linear view. */
    showShuffle: boolean;
    /** The legacy editor's per-element settings, preserved verbatim; `showShuffle` is mirrored into it. */
    settings?: Record<string, unknown>;
  };
}

export interface HeadingFormElement {
  id: string;
  type: 'heading';
  parentId?: string;
  data: {
    content: string;
  };
}

export interface TextFormElement {
  id: string;
  type: 'text';
  parentId?: string;
  data: {
    content: string;
  };
}

export interface DividerFormElement {
  id: string;
  type: 'divider';
  parentId?: string;
}

export type WorkflowFormElement =
  | ContainerFormElement
  | NodeFieldFormElement
  | HeadingFormElement
  | TextFormElement
  | DividerFormElement;

/** The Linear UI description: a tree of form elements rooted in a column container. */
export interface WorkflowForm {
  rootElementId: string;
  elements: Record<string, WorkflowFormElement>;
}

export interface WorkflowMetadata {
  name: string;
  description: string;
  author: string;
  contact: string;
  tags: string;
  notes: string;
  /** The workflow's own semver, distinct from the document schema version. */
  workflowVersion: string;
}

/** The project-owned workflow document. `version: 2` distinguishes it from the Phase-1 placeholder graph. */
export interface ProjectGraphState extends WorkflowMetadata {
  id: string;
  version: 2;
  /** Backend workflow-library binding when the document was loaded from or saved to the library. */
  libraryWorkflowId?: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  form: WorkflowForm;
  updatedAt: string;
}

export interface InvocationTemplatesSnapshot {
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
  templates: InvocationTemplates;
}

export const isInvocationNode = (node: WorkflowNode): node is WorkflowInvocationNode => node.type === 'invocation';

export const isNotesNode = (node: WorkflowNode): node is WorkflowNotesNode => node.type === 'notes';

export const isCurrentImageNode = (node: WorkflowNode): node is WorkflowCurrentImageNode =>
  node.type === 'current_image';

export const isConnectorNode = (node: WorkflowNode): node is WorkflowConnectorNode => node.type === 'connector';
