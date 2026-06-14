/**
 * Workflow domain types.
 *
 * The project graph is a *document*: an editable node-and-edge workflow plus a
 * form describing its Linear UI. It compiles into the queue-facing
 * `GraphContract` only at invocation time, so queued snapshots stay immutable
 * while the document keeps evolving. The serialized shape stays compatible
 * with the legacy WorkflowV3 format so workflows round-trip between the v6
 * editor, the workflow library backend, and this workbench.
 */

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
  /** How the field receives data: only via edge, only direct value, or either. */
  input: 'connection' | 'direct' | 'any';
  default?: unknown;
  uiHidden: boolean;
  uiOrder: number | null;
  uiComponent: 'slider' | 'textarea' | null;
  uiChoiceLabels: Record<string, string> | null;
  /** Enum choices when the field is an EnumField. */
  options: string[] | null;
  minimum: number | null;
  maximum: number | null;
  exclusiveMinimum: number | null;
  exclusiveMaximum: number | null;
  multipleOf: number | null;
  uiModelBase: string[] | null;
  uiModelType: string[] | null;
}

export interface FieldOutputTemplate {
  name: string;
  title: string;
  description: string;
  type: FieldType;
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
  /** User override of the template's field description (shown in the Linear UI). */
  description?: string;
  value?: unknown;
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

export type WorkflowNode = WorkflowInvocationNode | WorkflowNotesNode | WorkflowCurrentImageNode;

export interface WorkflowEdge {
  id: string;
  type: 'default';
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

export const isInvocationNode = (node: WorkflowNode): node is WorkflowInvocationNode => node.type === 'invocation';

export const isNotesNode = (node: WorkflowNode): node is WorkflowNotesNode => node.type === 'notes';

export const isCurrentImageNode = (node: WorkflowNode): node is WorkflowCurrentImageNode =>
  node.type === 'current_image';
