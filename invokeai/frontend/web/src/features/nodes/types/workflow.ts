import { nanoid } from 'nanoid';
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
});
export type WorkflowV3 = z.infer<typeof zWorkflowV3>;
// #endregion

// #region Workflow Builder

const zElementBase = z.object({
  id: z.string().trim().min(1),
});

const zNodeFieldElement = zElementBase.extend({
  type: z.literal('node-field'),
  data: z.object({
    fieldIdentifier: zFieldIdentifier,
  }),
});
export type NodeFieldElement = z.infer<typeof zNodeFieldElement>;
const nodeField = (
  nodeId: NodeFieldElement['data']['fieldIdentifier']['nodeId'],
  fieldName: NodeFieldElement['data']['fieldIdentifier']['fieldName']
): NodeFieldElement => ({
  id: nanoid(),
  type: 'node-field',
  data: {
    fieldIdentifier: { nodeId, fieldName },
  },
});

const zHeadingElement = zElementBase.extend({
  type: z.literal('heading'),
  data: z.object({
    content: z.string(),
    level: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4), z.literal(5)]),
  }),
});
export type HeadingElement = z.infer<typeof zHeadingElement>;
const heading = (
  content: HeadingElement['data']['content'],
  level: HeadingElement['data']['level']
): HeadingElement => ({
  id: nanoid(),
  type: 'heading',
  data: {
    content,
    level,
  },
});

const zTextElement = zElementBase.extend({
  type: z.literal('text'),
  data: z.object({
    content: z.string(),
    fontSize: z.enum(['sm', 'md', 'lg']),
  }),
});
export type TextElement = z.infer<typeof zTextElement>;
const text = (content: TextElement['data']['content'], fontSize: TextElement['data']['fontSize']): TextElement => ({
  id: nanoid(),
  type: 'text',
  data: {
    content,
    fontSize,
  },
});

const zDividerElement = zElementBase.extend({
  type: z.literal('divider'),
});
export type DividerElement = z.infer<typeof zDividerElement>;
const divider = (): DividerElement => ({
  id: nanoid(),
  type: 'divider',
});

export type ColumnElement = {
  id: string;
  type: 'column';
  data: {
    elements: ColumnChildElement[];
  };
};

const zColumnElement = zElementBase.extend({
  type: z.literal('column'),
  data: z.object({
    elements: z.lazy(() => z.array(zColumnChildElement)),
  }),
});
const column = (elements: ColumnElement['data']['elements']): ColumnElement => ({
  id: nanoid(),
  type: 'column',
  data: {
    elements,
  },
});

export type ContainerElement = {
  id: string;
  type: 'container';
  data: {
    columns: ColumnElement[];
  };
};

const zContainerElement: z.ZodType<ContainerElement> = zElementBase.extend({
  type: z.literal('container'),
  data: z.object({
    columns: z.lazy(() => z.array(zColumnElement)),
  }),
});
const container = (columns: ContainerElement['data']['columns']): ContainerElement => ({
  id: nanoid(),
  type: 'container',
  data: {
    columns,
  },
});

// export type CollapsibleElement = {
//   id: string;
//   type: 'collapsible';
//   columns: BuilderElement[];
//   title: string;
//   collapsed: boolean;
// };

// const zCollapsibleElement: z.ZodType<CollapsibleElement> = z.object({
//   type: z.literal('collapsible'),
//   columns: z.lazy(() => z.array(zElement)),
//   title: z.string(),
//   collapsed: z.boolean(),
// });

const zColumnChildElement = z.union([
  zContainerElement,
  // zCollapsibleElement
  zNodeFieldElement,
  zHeadingElement,
  zTextElement,
  zDividerElement,
]);

export type ColumnChildElement = z.infer<typeof zColumnChildElement>;

export const data: ContainerElement = container([
  column([
    heading('My Cool Workflow', 1),
    text('This is a description of what my workflow does. It does things.', 'md'),
    divider(),
    heading('First Section', 2),
    text('The first section includes fields relevant to the first section. This note describes that fact.', 'sm'),
    container([
      column([nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image')]),
      column([nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image')]),
      column([nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image')]),
    ]),
    nodeField('9c058600-8d73-4702-912b-0ccf37403bfd', 'value'),
    nodeField('7a8bbab2-6919-4cfc-bd7c-bcfda3c79ecf', 'value'),
    nodeField('4e16cbf6-457c-46fb-9ab7-9cb262fa1e03', 'value'),
    nodeField('39cb5272-a9d7-4da9-9c35-32e02b46bb34', 'color'),
    container([
      column([
        nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value'),
        nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value'),
      ]),
      column([
        nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value'),
        nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value'),
        nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value'),
        nodeField('4f609a81-0e25-47d1-ba0d-f24fedd5273f', 'value'),
      ]),
    ]),
    nodeField('14744f68-9000-4694-b4d6-cbe83ee231ee', 'model'),
    divider(),
    text('These are some text that are definitely super helpful.', 'sm'),
    divider(),
    container([
      column([
        nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image'),
        nodeField('7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', 'image'),
      ]),
      column([nodeField('7a8bbab2-6919-4cfc-bd7c-bcfda3c79ecf', 'value')]),
    ]),
  ]),
]);
