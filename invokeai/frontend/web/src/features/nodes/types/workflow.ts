import { getPrefixedId } from 'features/controlLayers/konva/util';
import { z } from 'zod';

import type { FieldType } from './field';
import { zFieldIdentifier } from './field';
import { zInvocationNodeData, zNotesNodeData } from './invocation';

// #region Workflow misc
const zXYPosition = z
  .object({
    x: z.number(),
    y: z.number(),
  })
  .default({ x: 0, y: 0 });
export type XYPosition = z.infer<typeof zXYPosition>;

const zWorkflowCategory = z.enum(['user', 'default', 'project']);
export type WorkflowCategory = z.infer<typeof zWorkflowCategory>;
// #endregion

// #region Workflow Nodes
const zWorkflowInvocationNode = z.object({
  id: z.string().trim().min(1),
  type: z.literal('invocation'),
  data: zInvocationNodeData,
  position: zXYPosition,
});
const zWorkflowNotesNode = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  data: zNotesNodeData,
  position: zXYPosition,
});
const zWorkflowNode = z.union([zWorkflowInvocationNode, zWorkflowNotesNode]);

type WorkflowInvocationNode = z.infer<typeof zWorkflowInvocationNode>;

export const isWorkflowInvocationNode = (val: unknown): val is WorkflowInvocationNode =>
  zWorkflowInvocationNode.safeParse(val).success;
// #endregion

// #region Workflow Edges
const zWorkflowEdgeBase = z.object({
  id: z.string().trim().min(1),
  source: z.string().trim().min(1),
  target: z.string().trim().min(1),
});
const zWorkflowEdgeDefault = zWorkflowEdgeBase.extend({
  type: z.literal('default'),
  sourceHandle: z.string().trim().min(1),
  targetHandle: z.string().trim().min(1),
  hidden: z.boolean().optional(),
});
const zWorkflowEdgeCollapsed = zWorkflowEdgeBase.extend({
  type: z.literal('collapsed'),
});
const zWorkflowEdge = z.union([zWorkflowEdgeDefault, zWorkflowEdgeCollapsed]);
// #endregion

// #region Workflow Builder
const zElementId = z.string().trim().min(1);
export type ElementId = z.infer<typeof zElementId>;

const zElementBase = z.object({
  id: zElementId,
  parentId: zElementId.optional(),
  data: z.undefined(),
});

export const zNumberComponent = z.enum(['number-input', 'slider', 'number-input-and-slider']);

const NODE_FIELD_TYPE = 'node-field';
export const NODE_FIELD_CLASS_NAME = getPrefixedId(NODE_FIELD_TYPE, '-');
const FLOAT_FIELD_SETTINGS_TYPE = 'float-field-config';
const zNodeFieldFloatSettings = z.object({
  type: z.literal(FLOAT_FIELD_SETTINGS_TYPE).default(FLOAT_FIELD_SETTINGS_TYPE),
  component: zNumberComponent.default('number-input'),
});
export const getFloatFieldSettingsDefaults = (): NodeFieldFloatSettings => zNodeFieldFloatSettings.parse({});
export type NodeFieldFloatSettings = z.infer<typeof zNodeFieldFloatSettings>;

const INTEGER_FIELD_CONFIG_TYPE = 'integer-field-config';
const zNodeFieldIntegerSettings = z.object({
  type: z.literal(INTEGER_FIELD_CONFIG_TYPE).default(INTEGER_FIELD_CONFIG_TYPE),
  component: zNumberComponent.default('number-input'),
});
export type NodeFieldIntegerSettings = z.infer<typeof zNodeFieldIntegerSettings>;
export const getIntegerFieldSettingsDefaults = (): NodeFieldIntegerSettings => zNodeFieldIntegerSettings.parse({});

export const zStringComponent = z.enum(['input', 'textarea']);
const STRING_FIELD_CONFIG_TYPE = 'string-field-config';
const zNodeFieldStringSettings = z.object({
  type: z.literal(STRING_FIELD_CONFIG_TYPE).default(STRING_FIELD_CONFIG_TYPE),
  component: zStringComponent.default('input'),
});
export type NodeFieldStringSettings = z.infer<typeof zNodeFieldStringSettings>;
export const getStringFieldSettingsDefaults = (): NodeFieldStringSettings => zNodeFieldStringSettings.parse({});

const zNodeFieldData = z.object({
  fieldIdentifier: zFieldIdentifier,
  showDescription: z.boolean().default(true),
  settings: z.union([zNodeFieldFloatSettings, zNodeFieldIntegerSettings, zNodeFieldStringSettings]).optional(),
});
const zNodeFieldElement = zElementBase.extend({
  type: z.literal(NODE_FIELD_TYPE),
  data: zNodeFieldData,
});
export type NodeFieldElement = z.infer<typeof zNodeFieldElement>;
export const isNodeFieldElement = (el: FormElement): el is NodeFieldElement => el.type === NODE_FIELD_TYPE;
export const buildNodeFieldElement = (
  nodeId: NodeFieldElement['data']['fieldIdentifier']['nodeId'],
  fieldName: NodeFieldElement['data']['fieldIdentifier']['fieldName'],
  fieldType: FieldType,
  parentId?: NodeFieldElement['parentId']
): NodeFieldElement => {
  let settings: NodeFieldElement['data']['settings'] = undefined;

  if (fieldType.name === 'IntegerField' && fieldType.cardinality === 'SINGLE') {
    settings = getIntegerFieldSettingsDefaults();
  }

  if (fieldType.name === 'FloatField' && fieldType.cardinality === 'SINGLE') {
    settings = getFloatFieldSettingsDefaults();
  }

  if (fieldType.name === 'StringField' && fieldType.cardinality === 'SINGLE') {
    settings = getStringFieldSettingsDefaults();
  }

  const element: NodeFieldElement = {
    id: getPrefixedId(NODE_FIELD_TYPE, '-'),
    type: NODE_FIELD_TYPE,
    parentId,
    data: {
      fieldIdentifier: { nodeId, fieldName },
      settings,
      showDescription: true,
    },
  };
  return element;
};

const HEADING_TYPE = 'heading';
export const HEADING_CLASS_NAME = getPrefixedId(HEADING_TYPE, '-');
const zHeadingElement = zElementBase.extend({
  type: z.literal(HEADING_TYPE),
  data: z.object({ content: z.string() }),
});
export type HeadingElement = z.infer<typeof zHeadingElement>;
export const isHeadingElement = (el: FormElement): el is HeadingElement => el.type === HEADING_TYPE;
export const buildHeading = (
  content: HeadingElement['data']['content'],
  parentId?: NodeFieldElement['parentId']
): HeadingElement => {
  const element: HeadingElement = {
    id: getPrefixedId(HEADING_TYPE, '-'),
    parentId,
    type: HEADING_TYPE,
    data: { content },
  };
  return element;
};

const TEXT_TYPE = 'text';
export const TEXT_CLASS_NAME = getPrefixedId(TEXT_TYPE, '-');
const zTextElement = zElementBase.extend({
  type: z.literal(TEXT_TYPE),
  data: z.object({ content: z.string() }),
});
export type TextElement = z.infer<typeof zTextElement>;
export const isTextElement = (el: FormElement): el is TextElement => el.type === TEXT_TYPE;
export const buildText = (
  content: TextElement['data']['content'],
  parentId?: NodeFieldElement['parentId']
): TextElement => {
  const element: TextElement = {
    id: getPrefixedId(TEXT_TYPE, '-'),
    parentId,
    type: TEXT_TYPE,
    data: { content },
  };
  return element;
};

const DIVIDER_TYPE = 'divider';
export const DIVIDER_CLASS_NAME = getPrefixedId(DIVIDER_TYPE, '-');
const zDividerElement = zElementBase.extend({
  type: z.literal(DIVIDER_TYPE),
});
export type DividerElement = z.infer<typeof zDividerElement>;
export const isDividerElement = (el: FormElement): el is DividerElement => el.type === DIVIDER_TYPE;
export const buildDivider = (parentId?: NodeFieldElement['parentId']): DividerElement => {
  const element: DividerElement = {
    id: getPrefixedId(DIVIDER_TYPE, '-'),
    parentId,
    type: DIVIDER_TYPE,
  };
  return element;
};

const CONTAINER_TYPE = 'container';
export const CONTAINER_CLASS_NAME = getPrefixedId(CONTAINER_TYPE, '-');
const zContainerElement = zElementBase.extend({
  type: z.literal(CONTAINER_TYPE),
  data: z.object({
    layout: z.enum(['row', 'column']),
    children: z.array(zElementId),
  }),
});
export type ContainerElement = z.infer<typeof zContainerElement>;
export const isContainerElement = (el: FormElement): el is ContainerElement => el.type === CONTAINER_TYPE;
export const buildContainer = (
  layout: ContainerElement['data']['layout'],
  children: ContainerElement['data']['children'],
  parentId?: NodeFieldElement['parentId']
): ContainerElement => {
  const element: ContainerElement = {
    id: getPrefixedId(CONTAINER_TYPE, '-'),
    parentId,
    type: CONTAINER_TYPE,
    data: {
      layout,
      children,
    },
  };
  return element;
};

const zFormElement = z.union([zContainerElement, zNodeFieldElement, zHeadingElement, zTextElement, zDividerElement]);

export type FormElement = z.infer<typeof zFormElement>;

export const getDefaultForm = () => {
  const rootElement = buildContainer('column', []);
  return {
    elements: {
      [rootElement.id]: rootElement,
    },
    rootElementId: rootElement.id,
  };
};

const zBuilderForm = z
  .object({
    elements: z.record(zFormElement),
    rootElementId: zElementId,
  })
  .default(getDefaultForm);
export type BuilderForm = z.infer<typeof zBuilderForm>;
//# endregion

// #region Workflow
export const zWorkflowV3 = z.object({
  id: z.string().min(1).optional(),
  name: z.string(),
  author: z.string(),
  description: z.string(),
  version: z.string(),
  contact: z.string(),
  tags: z.string(),
  notes: z.string(),
  nodes: z.array(zWorkflowNode),
  edges: z.array(zWorkflowEdge),
  exposedFields: z.array(zFieldIdentifier),
  meta: z.object({
    category: zWorkflowCategory.default('user'),
    version: z.literal('3.0.0'),
  }),
  form: zBuilderForm,
});
export type WorkflowV3 = z.infer<typeof zWorkflowV3>;
// #endregion
