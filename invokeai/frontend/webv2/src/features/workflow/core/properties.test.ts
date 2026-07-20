import { describe, expect, it } from 'vitest';

import type { InvocationTemplate, ProjectGraphState, WorkflowEdge } from './types';

import {
  buildInvocationNode,
  createProjectGraph,
  getFormChildren,
  isFieldExposed,
  projectGraphReducer,
} from './document';
import { parseWorkflowJson, serializeWorkflowJson } from './workflowJson';

const template: InvocationTemplate = {
  category: 'property-test',
  classification: 'stable',
  description: '',
  inputs: {
    value: {
      default: '',
      description: '',
      exclusiveMaximum: null,
      exclusiveMinimum: null,
      input: 'any',
      maximum: null,
      minimum: null,
      multipleOf: null,
      name: 'value',
      options: null,
      required: false,
      title: 'Value',
      type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
      uiChoiceLabels: null,
      uiComponent: null,
      uiHidden: false,
      uiModelBase: null,
      uiModelType: null,
      uiOrder: null,
    },
  },
  nodePack: 'invokeai',
  outputs: {
    value: {
      description: '',
      name: 'value',
      title: 'Value',
      type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
    },
  },
  outputType: 'string_output',
  tags: [],
  title: 'Value',
  type: 'value',
  useCache: true,
  version: '1.0.0',
};

const buildScenario = (seed: number): ProjectGraphState => {
  const nodeCount = (seed % 6) + 1;
  let document = createProjectGraph(`property-${seed}`, `Workflow ${seed}`);

  for (let index = 0; index < nodeCount; index += 1) {
    const node = buildInvocationNode(template, { x: seed * 3 + index * 17, y: index * 11 });

    document = projectGraphReducer(document, { node, type: 'addNode' });
    document = projectGraphReducer(document, {
      fieldName: 'value',
      nodeId: node.id,
      type: 'setFieldValue',
      value: `seed-${seed}-node-${index}`,
    });

    if ((seed + index) % 2 === 0) {
      document = projectGraphReducer(document, {
        fieldIdentifier: { fieldName: 'value', nodeId: node.id },
        type: 'exposeField',
      });
    }
  }

  for (let index = 1; index < document.nodes.length; index += 1) {
    const source = document.nodes[index - 1];
    const target = document.nodes[index];

    if (source && target) {
      const edge: WorkflowEdge = {
        id: `edge-${seed}-${index}`,
        source: source.id,
        sourceHandle: 'value',
        target: target.id,
        targetHandle: 'value',
        type: 'default',
      };

      document = projectGraphReducer(document, { edge, type: 'addEdge' });
    }
  }

  return document;
};

describe('workflow document properties', () => {
  it('keeps transport round-trips, migrations, edges, and linear projections stable across varied documents', () => {
    for (let seed = 0; seed < 128; seed += 1) {
      const original = buildScenario(seed);
      const serialized = serializeWorkflowJson(original);
      const firstParse = parseWorkflowJson(serialized);
      const secondParse = parseWorkflowJson(serializeWorkflowJson(firstParse.document));

      expect(firstParse.warnings, `first parse warnings for seed ${seed}`).toEqual([]);
      expect(secondParse.warnings, `second parse warnings for seed ${seed}`).toEqual([]);
      expect(serializeWorkflowJson(secondParse.document), `stable migration for seed ${seed}`).toEqual(
        serializeWorkflowJson(firstParse.document)
      );

      const nodeIds = new Set(firstParse.document.nodes.map((node) => node.id));

      expect(
        firstParse.document.edges.every((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)),
        `edge validity for seed ${seed}`
      ).toBe(true);

      const projectedFields = getFormChildren(firstParse.document.form).filter(
        (element) => element.type === 'node-field'
      );

      for (const element of projectedFields) {
        expect(nodeIds.has(element.data.fieldIdentifier.nodeId), `projection node for seed ${seed}`).toBe(true);
        expect(
          isFieldExposed(firstParse.document.form, element.data.fieldIdentifier),
          `projection exposure for seed ${seed}`
        ).toBe(true);
      }
    }
  });
});
