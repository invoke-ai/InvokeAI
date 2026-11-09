import type { InvocationTemplate, ProjectGraphState } from '@features/workflow/core/types';
import type { WorkflowLibraryEntryEnrichment } from '@features/workflow/data/libraryBrowseStore';

import { buildInvocationNode, createProjectGraph } from '@features/workflow/utility';
import { describe, expect, it } from 'vitest';

import { getMissingNodePacks } from './WorkflowRequirementsList';

const template = (type: string, nodePack: string): InvocationTemplate => ({
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {},
  nodePack,
  outputs: {},
  outputType: 'integer_output',
  tags: [],
  title: type,
  type,
  useCache: true,
  version: '1.0.0',
});

const ready = (document: ProjectGraphState): Extract<WorkflowLibraryEntryEnrichment, { status: 'ready' }> => ({
  document,
  nodeCount: document.nodes.length,
  requirements: { primaryBase: null, requirements: [] },
  status: 'ready',
});

describe('getMissingNodePacks', () => {
  const known = template('integer', 'invokeai');
  const document: ProjectGraphState = {
    ...createProjectGraph('packs'),
    nodes: [
      buildInvocationNode(known, { x: 0, y: 0 }),
      buildInvocationNode(template('face_mask', 'face-tools'), { x: 400, y: 0 }),
      buildInvocationNode(template('face_crop', 'face-tools'), { x: 800, y: 0 }),
      buildInvocationNode(template('wobble', 'wobbly'), { x: 1200, y: 0 }),
    ],
  };

  it('names each pack whose node types the server lacks, once', () => {
    expect(getMissingNodePacks(ready(document), { integer: known })).toEqual(['face-tools', 'wobbly']);
  });

  it('answers from cache for the same enrichment and templates, and recomputes for new templates', () => {
    const enrichment = ready(document);
    const templates = { integer: known };
    const first = getMissingNodePacks(enrichment, templates);

    expect(getMissingNodePacks(enrichment, templates)).toBe(first);
    expect(
      getMissingNodePacks(enrichment, {
        ...templates,
        face_crop: template('face_crop', 'face-tools'),
        face_mask: template('face_mask', 'face-tools'),
        wobble: template('wobble', 'wobbly'),
      })
    ).toEqual([]);
  });
});
