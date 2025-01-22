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
const buildElementBuilder =
  <T extends BuilderElement['type']>(type: T) =>
  (data: Extract<BuilderElement, { type: T }>['data']): Extract<BuilderElement, { type: T }> =>
    ({
      id: nanoid(),
      type,
      data,
    }) as Extract<BuilderElement, { type: T }>;
const zFieldElementNumberConfig = z.object({
  display: z.enum(['slider', 'input', 'input-with-slider-inline', 'input-with-slider-popover']),
});
const zFieldElementStringConfig = z.object({
  display: z.enum(['input', 'textarea']),
});

const zElementBase = z.object({
  id: z.string().trim().min(1),
});

const zFieldElement = zElementBase.extend({
  type: z.literal('field'),
  data: z.object({
    fieldIdentifier: zFieldIdentifier,
    fieldConfig: z.union([zFieldElementNumberConfig, zFieldElementStringConfig]).optional(),
  }),
});
export type FieldElement = z.infer<typeof zFieldElement>;
const buildFieldElement = buildElementBuilder('field');

const zHeadingElement = zElementBase.extend({
  type: z.literal('heading'),
  data: z.object({
    content: z.string(),
    level: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4), z.literal(5)]),
  }),
});
export type HeadingElement = z.infer<typeof zHeadingElement>;
const buildHeadingElement = buildElementBuilder('heading');

const zTextElement = zElementBase.extend({
  type: z.literal('text'),
  data: z.object({
    content: z.string(),
    fontSize: z.enum(['sm', 'md', 'lg']),
  }),
});
export type TextElement = z.infer<typeof zTextElement>;
const buildTextElement = buildElementBuilder('text');

const zDividerElement = zElementBase.extend({
  type: z.literal('divider'),
  data: z.void(),
});
export type DividerElement = z.infer<typeof zDividerElement>;
const buildDividerElement = buildElementBuilder('divider');

export type StackElement = {
  id: string;
  type: 'stack';
  data: {
    children: BuilderElement[];
    direction: 'horizontal' | 'vertical';
  };
};

const zStackElement: z.ZodType<StackElement> = zElementBase.extend({
  type: z.literal('stack'),
  data: z.object({
    children: z.lazy(() => z.array(zElement)),
    direction: z.enum(['horizontal', 'vertical']),
  }),
});
const buildStackElement = buildElementBuilder('stack');

// export type CollapsibleElement = {
//   id: string;
//   type: 'collapsible';
//   children: BuilderElement[];
//   title: string;
//   collapsed: boolean;
// };

// const zCollapsibleElement: z.ZodType<CollapsibleElement> = z.object({
//   type: z.literal('collapsible'),
//   children: z.lazy(() => z.array(zElement)),
//   title: z.string(),
//   collapsed: z.boolean(),
// });

const zElement = z.union([
  zStackElement,
  // zCollapsibleElement
  zFieldElement,
  zHeadingElement,
  zTextElement,
  zDividerElement,
]);

export type BuilderElement =
  | StackElement
  // | CollapsibleElement
  | FieldElement
  | HeadingElement
  | TextElement
  | DividerElement;

export const data: StackElement = buildStackElement({
  direction: 'vertical',
  children: [
    buildHeadingElement({ content: 'My Cool Workflow', level: 1 }),
    buildTextElement({ content: 'This is a description of what my workflow does. It does things.', fontSize: 'md' }),
    buildDividerElement(),
    buildHeadingElement({ content: 'First Section', level: 2 }),
    buildTextElement({
      content: 'The first section includes fields relevant to the first section. This note describes that fact.',
      fontSize: 'sm',
    }),
    buildStackElement({
      direction: 'horizontal',
      children: [
        buildFieldElement({
          fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
        }),
        buildFieldElement({
          fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
        }),
        buildFieldElement({
          fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
        }),
      ],
    }),
    buildFieldElement({ fieldIdentifier: { nodeId: '9c058600-8d73-4702-912b-0ccf37403bfd', fieldName: 'value' } }),
    buildFieldElement({ fieldIdentifier: { nodeId: '7a8bbab2-6919-4cfc-bd7c-bcfda3c79ecf', fieldName: 'value' } }),
    buildFieldElement({ fieldIdentifier: { nodeId: '4e16cbf6-457c-46fb-9ab7-9cb262fa1e03', fieldName: 'value' } }),
    buildFieldElement({ fieldIdentifier: { nodeId: '39cb5272-a9d7-4da9-9c35-32e02b46bb34', fieldName: 'color' } }),
    buildStackElement({
      direction: 'horizontal',
      children: [
        buildFieldElement({
          fieldIdentifier: { nodeId: '4f609a81-0e25-47d1-ba0d-f24fedd5273f', fieldName: 'value' },
        }),
        buildFieldElement({
          fieldIdentifier: { nodeId: '4f609a81-0e25-47d1-ba0d-f24fedd5273f', fieldName: 'value' },
        }),
        buildFieldElement({
          fieldIdentifier: { nodeId: '4f609a81-0e25-47d1-ba0d-f24fedd5273f', fieldName: 'value' },
        }),
        buildFieldElement({
          fieldIdentifier: { nodeId: '4f609a81-0e25-47d1-ba0d-f24fedd5273f', fieldName: 'value' },
        }),
      ],
    }),
    buildFieldElement({ fieldIdentifier: { nodeId: '14744f68-9000-4694-b4d6-cbe83ee231ee', fieldName: 'model' } }),
    buildDividerElement(),
    buildTextElement({ content: 'These are some text that are definitely super helpful.', fontSize: 'sm' }),
    buildDividerElement(),
    buildStackElement({
      direction: 'horizontal',
      children: [
        buildStackElement({
          direction: 'vertical',
          children: [
            buildFieldElement({
              fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
            }),
            buildFieldElement({
              fieldIdentifier: { nodeId: '7aed1a5f-7fd7-4184-abe8-ddea0ea5e706', fieldName: 'image' },
            }),
          ],
        }),
        buildDividerElement(),
        buildFieldElement({
          fieldIdentifier: { nodeId: '7a8bbab2-6919-4cfc-bd7c-bcfda3c79ecf', fieldName: 'value' },
        }),
      ],
    }),
  ],
});
