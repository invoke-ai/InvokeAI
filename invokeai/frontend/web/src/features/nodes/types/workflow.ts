import { getPrefixedId } from 'features/controlLayers/konva/util';
import { z } from 'zod';

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
  form: z
    .object({
      elements: z.record(z.lazy(() => zFormElement)),
      rootElementId: z.lazy(() => zElementId),
    })
    .optional(),
});
export type WorkflowV3 = z.infer<typeof zWorkflowV3>;
// #endregion

// #region Workflow Builder
export const elements: Record<string, FormElement> = {};

export const addElement = (element: FormElement) => {
  elements[element.id] = element;
};

const zElementId = z.string().trim().min(1);
export type ElementId = z.infer<typeof zElementId>;

const zElementBase = z.object({
  id: zElementId,
  parentId: zElementId.optional(),
});

const NODE_FIELD_TYPE = 'node-field';
export const NODE_FIELD_CLASS_NAME = getPrefixedId(NODE_FIELD_TYPE, '-');
const zNodeFieldElement = zElementBase.extend({
  type: z.literal(NODE_FIELD_TYPE),
  data: z.object({
    fieldIdentifier: zFieldIdentifier,
  }),
});
export type NodeFieldElement = z.infer<typeof zNodeFieldElement>;
export const isNodeFieldElement = (el: FormElement): el is NodeFieldElement => el.type === NODE_FIELD_TYPE;
export const buildNodeField = (
  nodeId: NodeFieldElement['data']['fieldIdentifier']['nodeId'],
  fieldName: NodeFieldElement['data']['fieldIdentifier']['fieldName'],
  parentId?: NodeFieldElement['parentId']
): NodeFieldElement => {
  const element: NodeFieldElement = {
    id: getPrefixedId(NODE_FIELD_TYPE, '-'),
    type: NODE_FIELD_TYPE,
    parentId,
    data: {
      fieldIdentifier: { nodeId, fieldName },
    },
  };
  return element;
};
const _nodeField = (...args: Parameters<typeof buildNodeField>): NodeFieldElement => {
  const element = buildNodeField(...args);
  addElement(element);
  return element;
};

const HEADING_TYPE = 'heading';
export const HEADING_CLASS_NAME = getPrefixedId(HEADING_TYPE, '-');
const zHeadingElement = zElementBase.extend({
  type: z.literal(HEADING_TYPE),
  data: z.object({
    content: z.string(),
    level: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4), z.literal(5)]),
  }),
});
export type HeadingElement = z.infer<typeof zHeadingElement>;
export const isHeadingElement = (el: FormElement): el is HeadingElement => el.type === HEADING_TYPE;
export const buildHeading = (
  content: HeadingElement['data']['content'],
  level: HeadingElement['data']['level'],
  parentId?: NodeFieldElement['parentId']
): HeadingElement => {
  const element: HeadingElement = {
    id: getPrefixedId(HEADING_TYPE, '-'),
    parentId,
    type: HEADING_TYPE,
    data: {
      content,
      level,
    },
  };
  return element;
};
const _heading = (...args: Parameters<typeof buildHeading>): HeadingElement => {
  const element = buildHeading(...args);
  addElement(element);
  return element;
};

const TEXT_TYPE = 'text';
export const TEXT_CLASS_NAME = getPrefixedId(TEXT_TYPE, '-');
const zTextElement = zElementBase.extend({
  type: z.literal(TEXT_TYPE),
  data: z.object({
    content: z.string(),
    fontSize: z.enum(['sm', 'md', 'lg']),
  }),
});
export type TextElement = z.infer<typeof zTextElement>;
export const isTextElement = (el: FormElement): el is TextElement => el.type === TEXT_TYPE;
export const buildText = (
  content: TextElement['data']['content'],
  fontSize: TextElement['data']['fontSize'],
  parentId?: NodeFieldElement['parentId']
): TextElement => {
  const element: TextElement = {
    id: getPrefixedId(TEXT_TYPE, '-'),
    parentId,
    type: TEXT_TYPE,
    data: {
      content,
      fontSize,
    },
  };
  return element;
};
const _text = (...args: Parameters<typeof buildText>): TextElement => {
  const element = buildText(...args);
  addElement(element);
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
const _divider = (...args: Parameters<typeof buildDivider>): DividerElement => {
  const element = buildDivider(...args);
  addElement(element);
  return element;
};

const CONTAINER_TYPE = 'container';
export const CONTAINER_CLASS_NAME = getPrefixedId(CONTAINER_TYPE, '-');
const zContainerElement = zElementBase.extend({
  type: z.literal(CONTAINER_TYPE),
  data: z.object({
    direction: z.enum(['row', 'column']),
    children: z.array(zElementId),
  }),
});
export type ContainerElement = z.infer<typeof zContainerElement>;
export const isContainerElement = (el: FormElement): el is ContainerElement => el.type === CONTAINER_TYPE;
export const buildContainer = (
  direction: ContainerElement['data']['direction'],
  children: ContainerElement['data']['children'],
  parentId?: NodeFieldElement['parentId']
): ContainerElement => {
  const element: ContainerElement = {
    id: getPrefixedId(CONTAINER_TYPE, '-'),
    parentId,
    type: CONTAINER_TYPE,
    data: {
      direction,
      children,
    },
  };
  return element;
};
export const _container = (...args: Parameters<typeof buildContainer>): ContainerElement => {
  const element = buildContainer(...args);
  addElement(element);
  return element;
};

const zFormElement = z.union([zContainerElement, zNodeFieldElement, zHeadingElement, zTextElement, zDividerElement]);

export type FormElement = z.infer<typeof zFormElement>;

// export const rootElementId: string = _container('column', [
//   _heading('My Cool Workflow', 1).id,
//   _nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value').id,
//   _nodeField('14744f68-9000-4694-b4d6-cbe83ee231ee', 'model').id,
//   _container('row', [_container('column', []).id, _container('column', []).id, _container('column', []).id]).id,
// ]).id;

const rootContainer = buildContainer('column', []);
addElement(rootContainer);
const rowContainer = buildContainer('row', [], rootContainer.id);
const rowContainerChildren = [
  _nodeField('58e748ec-7405-4816-a5ff-c168ee35161a', 'value', rowContainer.id),
  _nodeField('1b383334-2efc-406d-a000-49c4c5ebccde', 'value', rowContainer.id),
  _nodeField('7ebda150-63c7-4ccd-a1d2-444459107393', 'value', rowContainer.id),
];
rowContainerChildren.forEach((child) => {
  addElement(child);
  rowContainer.data.children.push(child.id);
});

const children = [
  buildHeading('My Cool Workflow', 1, rootContainer.id),
  buildText('This is a description of what my workflow does. It does things.', 'md', rootContainer.id),
  buildDivider(rootContainer.id),
  buildText('These are some text that are definitely super helpful.', 'sm', rootContainer.id),
  rowContainer,
];
children.forEach((child) => {
  addElement(child);
  rootContainer.data.children.push(child.id);
});

export const rootElementId = rootContainer.id;
// export const rootElementId: string = _container('column', [
//   _heading('My Cool Workflow', 1).id,
//   _text('This is a description of what my workflow does. It does things.', 'md').id,
//   _divider().id,
//   _heading('First Section', 2).id,
//   _text('The first section includes fields relevant to the first section. This note describes that fact.', 'sm').id,
//   _divider().id,
//   // _container('row', [
//   //   _nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image').id,
//   //   _nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image').id,
//   //   _nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image').id,
//   // ]).id,
//   // _nodeField('9c058600-8d73-4702-912b-0ccf37403bfd', 'value').id,
//   // _nodeField('7a8bbab2-6919-4cfc-bd7c-bcfda3c79ecf', 'value').id,
//   // _nodeField('4e16cbf6-457c-46fb-9ab7-9cb262fa1e03', 'value').id,
//   // _nodeField('39cb5272-a9d7-4da9-9c35-32e02b46bb34', 'color').id,
//   // _container('row', [
//   //   _container('column', [
//   //     _nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value').id,
//   //     _nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value').id,
//   //   ]).id,
//   //   _container('column', [
//   //     _nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value').id,
//   //     _nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value').id,
//   //     _nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image').id,
//   //     _nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value').id,
//   //   ]).id,
//   // ]).id,
//   // _nodeField('14744f68-9000-4694-b4d6-cbe83ee231ee', 'model').id,
//   _divider().id,
//   _text('These are some text that are definitely super helpful.', 'sm').id,
//   _divider().id,
//   // _container('row', [
//   //   _container('column', [
//   //     _nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image').id,
//   //     _nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image').id,
//   //   ]).id,
//   //   _container('column', [_nodeField('7a8bbab2-6919-4cfc-bd7c-bcfda3c79ecf', 'value').id]).id,
//   // ]).id,
// ]).id;
