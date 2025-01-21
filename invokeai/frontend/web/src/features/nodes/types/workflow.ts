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
const zElementId = z.string().trim().min(1);

const zFieldElement = z.object({
  id: zElementId,
  type: z.literal('field'),
  fieldIdentifier: zFieldIdentifier,
});
export type FieldElement = z.infer<typeof zFieldElement>;

const zHeadingElement = z.object({
  id: zElementId,
  type: z.literal('heading'),
  content: z.string(),
  level: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4), z.literal(5)]),
});
export type HeadingElement = z.infer<typeof zHeadingElement>;

const zNotesElement = z.object({
  id: zElementId,
  type: z.literal('notes'),
  content: z.string(),
  fontSize: z.enum(['sm', 'md', 'lg']),
});
export type NotesElement = z.infer<typeof zNotesElement>;

const zDividerElement = z.object({
  id: zElementId,
  type: z.literal('divider'),
});
export type DividerElement = z.infer<typeof zDividerElement>;

export type ContainerElement = {
  id: string;
  type: 'container';
  children: BuilderElement[];
  orientation: 'horizontal' | 'vertical';
};

export type BuilderElement = FieldElement | HeadingElement | NotesElement | DividerElement | ContainerElement;

const zContainerElement: z.ZodType<ContainerElement> = z.object({
  id: zElementId,
  type: z.literal('container'),
  children: z.lazy(() => z.array(zElement)),
  orientation: z.enum(['horizontal', 'vertical']),
});

const zElement = z.union([zFieldElement, zNotesElement, zDividerElement, zContainerElement]);

type ElementType = BuilderElement['type'];

export const data: ContainerElement = {
  id: nanoid(),
  type: 'container',
  orientation: 'vertical',
  children: [
    { id: nanoid(), type: 'heading', content: 'My Cool Workflow', level: 1 },
    {
      id: nanoid(),
      type: 'notes',
      content: 'This is a description of what my workflow does. It does things.',
      fontSize: 'md',
    },
    { id: nanoid(), type: 'heading', content: 'First Section', level: 2 },
    {
      id: nanoid(),
      type: 'notes',
      content: 'The first section includes fields relevant to the first section. This note describes that fact.',
      fontSize: 'sm',
    },
    {
      id: nanoid(),
      type: 'container',
      orientation: 'horizontal',
      children: [
        {
          id: nanoid(),
          type: 'field',
          fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
        },
        {
          id: nanoid(),
          type: 'field',
          fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
        },
        {
          id: nanoid(),
          type: 'field',
          fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
        },
      ],
    },
    {
      id: nanoid(),
      type: 'field',
      fieldIdentifier: { nodeId: '9c058600-8d73-4702-912b-0ccf37403bfd', fieldName: 'value' },
    },
    {
      id: nanoid(),
      type: 'field',
      fieldIdentifier: { nodeId: '7a8bbab2-6919-4cfc-bd7c-bcfda3c79ecf', fieldName: 'value' },
    },
    {
      id: nanoid(),
      type: 'field',
      fieldIdentifier: { nodeId: '4e16cbf6-457c-46fb-9ab7-9cb262fa1e03', fieldName: 'value' },
    },
    {
      id: nanoid(),
      type: 'field',
      fieldIdentifier: { nodeId: '39cb5272-a9d7-4da9-9c35-32e02b46bb34', fieldName: 'color' },
    },
    {
      id: nanoid(),
      type: 'container',
      orientation: 'horizontal',
      children: [
        {
          id: nanoid(),
          type: 'field',
          fieldIdentifier: { nodeId: '4f609a81-0e25-47d1-ba0d-f24fedd5273f', fieldName: 'value' },
        },
        {
          id: nanoid(),
          type: 'field',
          fieldIdentifier: { nodeId: '4f609a81-0e25-47d1-ba0d-f24fedd5273f', fieldName: 'value' },
        },
        {
          id: nanoid(),
          type: 'field',
          fieldIdentifier: { nodeId: '4f609a81-0e25-47d1-ba0d-f24fedd5273f', fieldName: 'value' },
        },
        {
          id: nanoid(),
          type: 'field',
          fieldIdentifier: { nodeId: '4f609a81-0e25-47d1-ba0d-f24fedd5273f', fieldName: 'value' },
        },
      ],
    },
    {
      id: nanoid(),
      type: 'field',
      fieldIdentifier: { nodeId: '14744f68-9000-4694-b4d6-cbe83ee231ee', fieldName: 'model' },
    },
    { id: nanoid(), type: 'divider' },
    { id: nanoid(), type: 'notes', content: 'These are some notes that are definitely super helpful.', fontSize: 'sm' },
    { id: nanoid(), type: 'divider' },
    {
      id: nanoid(),
      type: 'container',
      orientation: 'horizontal',
      children: [
        {
          id: nanoid(),
          type: 'container',
          orientation: 'vertical',
          children: [
            {
              id: nanoid(),
              type: 'field',
              fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
            },
            {
              id: nanoid(),
              type: 'field',
              fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
            },
          ],
        },
        { id: nanoid(), type: 'divider' },
        {
          id: nanoid(),
          type: 'field',
          fieldIdentifier: { nodeId: '7a8bbab2-6919-4cfc-bd7c-bcfda3c79ecf', fieldName: 'value' },
        },
      ],
    },
  ],
};
