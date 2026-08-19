import { describe, expect, it } from 'vitest';

import type {
  FieldInputTemplate,
  InvocationTemplate,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowEdge,
  WorkflowInvocationNode,
} from './types';

import {
  extractWorkflowModelRequirements,
  getAddModelsSearchTerm,
  resolveWorkflowModelRequirements,
  type InstalledModelSummary,
  type ResolvedModelRequirement,
  type StarterCatalogEntry,
  type WorkflowModelRequirement,
} from './modelRequirements';

/**
 * Fixture helpers mirror `graphToDocument.test.ts`: minimal templates and
 * documents built inline rather than mocking the module under test.
 */

const fieldInput = (name: string, overrides: Partial<FieldInputTemplate> = {}): FieldInputTemplate => ({
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required: false,
  title: name,
  type: { batch: false, cardinality: 'SINGLE', name: 'ModelIdentifierField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

const invocationTemplate = (type: string, inputs: Record<string, FieldInputTemplate>): InvocationTemplate => ({
  category: 'test',
  classification: 'stable',
  description: '',
  inputs,
  nodePack: 'invokeai',
  outputs: {},
  outputType: `${type}_output`,
  tags: [],
  title: type,
  type,
  useCache: true,
  version: '1.0.0',
});

const invocationNode = (id: string, type: string, inputs: Record<string, unknown> = {}): WorkflowInvocationNode => ({
  data: {
    inputs: Object.fromEntries(Object.entries(inputs).map(([name, value]) => [name, { label: '', name, value }])),
    isIntermediate: false,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type,
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x: 0, y: 0 },
  type: 'invocation',
});

const graphDocument = (nodes: WorkflowInvocationNode[], edges: WorkflowEdge[] = []): ProjectGraphState => ({
  author: '',
  contact: '',
  description: '',
  edges,
  form: { elements: {}, rootElementId: 'root' },
  id: 'doc',
  name: 'doc',
  nodes,
  notes: '',
  tags: '',
  updatedAt: '2024-01-01T00:00:00.000Z',
  version: 2,
  workflowVersion: '1.0.0',
});

describe('extractWorkflowModelRequirements', () => {
  it('extracts an exact requirement when a model field value duck-types a model identifier', () => {
    const templates: InvocationTemplates = {
      main_model_loader: invocationTemplate('main_model_loader', {
        model: fieldInput('model', { required: true }),
      }),
    };
    const document = graphDocument([
      invocationNode('loader', 'main_model_loader', {
        model: { base: 'sdxl', hash: 'hash-1', key: 'model-key-1', name: 'My Model', type: 'main' },
      }),
    ]);

    const { requirements } = extractWorkflowModelRequirements(document, templates);

    expect(requirements).toEqual([
      {
        identifier: { base: 'sdxl', hash: 'hash-1', key: 'model-key-1', name: 'My Model', type: 'main' },
        kind: 'exact',
        label: 'My Model',
      },
    ]);
  });

  it('skips inputs that are fed by a connection', () => {
    const templates: InvocationTemplates = {
      main_model_loader: invocationTemplate('main_model_loader', {
        model: fieldInput('model', { required: true, uiModelBase: ['sdxl'], uiModelType: ['main'] }),
      }),
      upstream: invocationTemplate('upstream', {}),
    };
    const document = graphDocument(
      [invocationNode('upstream', 'upstream'), invocationNode('loader', 'main_model_loader', {})],
      [
        {
          id: 'e1',
          source: 'upstream',
          sourceHandle: 'model',
          target: 'loader',
          targetHandle: 'model',
          type: 'default',
        },
      ]
    );

    const { requirements } = extractWorkflowModelRequirements(document, templates);

    expect(requirements).toEqual([]);
  });

  it('dedupes exact requirements that share a key', () => {
    const templates: InvocationTemplates = {
      loader: invocationTemplate('loader', { model: fieldInput('model', { required: true }) }),
    };
    const document = graphDocument([
      invocationNode('a', 'loader', { model: { key: 'shared-key', name: 'Shared' } }),
      invocationNode('b', 'loader', { model: { key: 'shared-key', name: 'Shared' } }),
    ]);

    const { requirements } = extractWorkflowModelRequirements(document, templates);

    expect(requirements).toHaveLength(1);
  });

  it('extracts a slot requirement for an empty required model field, labeled from uiModelBase/uiModelType', () => {
    const templates: InvocationTemplates = {
      main_model_loader: invocationTemplate('main_model_loader', {
        model: fieldInput('model', { required: true, title: 'Model', uiModelBase: ['sdxl'], uiModelType: ['main'] }),
      }),
    };
    const document = graphDocument([invocationNode('loader', 'main_model_loader', {})]);

    const { requirements } = extractWorkflowModelRequirements(document, templates);

    expect(requirements).toEqual([{ base: 'sdxl', kind: 'slot', label: 'SDXL checkpoint', modelType: 'main' }]);
  });

  it('does not extract a slot requirement for an empty optional model field', () => {
    const templates: InvocationTemplates = {
      main_model_loader: invocationTemplate('main_model_loader', {
        model: fieldInput('model', { required: false, uiModelBase: ['sdxl'], uiModelType: ['main'] }),
      }),
    };
    const document = graphDocument([invocationNode('loader', 'main_model_loader', {})]);

    const { requirements } = extractWorkflowModelRequirements(document, templates);

    expect(requirements).toEqual([]);
  });

  it('prefers the most frequent main-type base for primaryBase over a more frequent non-main base', () => {
    const templates: InvocationTemplates = {
      loader: invocationTemplate('loader', { model: fieldInput('model', { required: true }) }),
      vae_loader: invocationTemplate('vae_loader', { vae_model: fieldInput('vae_model', { required: true }) }),
    };
    const document = graphDocument([
      invocationNode('a', 'loader', { model: { base: 'sd-1', key: 'k1', type: 'main' } }),
      invocationNode('b', 'loader', { model: { base: 'sdxl', key: 'k2', type: 'main' } }),
      invocationNode('c', 'loader', { model: { base: 'sdxl', key: 'k3', type: 'main' } }),
      invocationNode('d', 'vae_loader', { vae_model: { base: 'sd-1', key: 'k4', type: 'vae' } }),
      invocationNode('e', 'vae_loader', { vae_model: { base: 'sd-1', key: 'k5', type: 'vae' } }),
    ]);

    const { primaryBase } = extractWorkflowModelRequirements(document, templates);

    expect(primaryBase).toBe('sdxl');
  });

  it('falls back to the most frequent base overall when no exact main-type identifier is present', () => {
    const templates: InvocationTemplates = {
      vae_loader: invocationTemplate('vae_loader', {
        vae_model: fieldInput('vae_model', { required: true, uiModelBase: ['flux'], uiModelType: ['vae'] }),
      }),
    };
    const document = graphDocument([invocationNode('a', 'vae_loader', {})]);

    const { primaryBase } = extractWorkflowModelRequirements(document, templates);

    expect(primaryBase).toBe('flux');
  });

  it('returns a null primaryBase when there are no requirements with a known base', () => {
    const document = graphDocument([]);

    const { primaryBase, requirements } = extractWorkflowModelRequirements(document, {});

    expect(requirements).toEqual([]);
    expect(primaryBase).toBeNull();
  });
});

describe('resolveWorkflowModelRequirements', () => {
  it('resolves an exact requirement as installed via a plain key hit', () => {
    const requirement: WorkflowModelRequirement = {
      identifier: { key: 'model-key-1', name: 'My Model' },
      kind: 'exact',
      label: 'My Model',
    };
    const installedModels: InstalledModelSummary[] = [
      { base: 'sdxl', hash: 'hash-1', key: 'model-key-1', name: 'My Model', type: 'main' },
    ];

    const [resolved] = resolveWorkflowModelRequirements([requirement], {
      activeInstallSources: new Set(),
      installedModels,
      starterModels: [],
    });

    expect(resolved).toMatchObject({ matchedModelName: 'My Model', starterMatch: null, status: 'installed' });
  });

  it('falls back to a hash match when the key does not hit', () => {
    const requirement: WorkflowModelRequirement = {
      identifier: { base: 'sdxl', hash: 'hash-1', key: 'unknown-key', name: 'My Model', type: 'main' },
      kind: 'exact',
      label: 'My Model',
    };
    const installedModels: InstalledModelSummary[] = [
      { base: 'sdxl', hash: 'hash-1', key: 'installed-key', name: 'My Model', type: 'main' },
    ];

    const [resolved] = resolveWorkflowModelRequirements([requirement], {
      activeInstallSources: new Set(),
      installedModels,
      starterModels: [],
    });

    expect(resolved).toMatchObject({ matchedModelName: 'My Model', status: 'installed' });
  });

  it('falls back to a name+base+type match when key and hash both miss', () => {
    const requirement: WorkflowModelRequirement = {
      identifier: { base: 'sdxl', hash: 'unknown-hash', key: 'unknown-key', name: 'My Model', type: 'main' },
      kind: 'exact',
      label: 'My Model',
    };
    const installedModels: InstalledModelSummary[] = [
      { base: 'sdxl', hash: 'different-hash', key: 'installed-key', name: 'My Model', type: 'main' },
    ];

    const [resolved] = resolveWorkflowModelRequirements([requirement], {
      activeInstallSources: new Set(),
      installedModels,
      starterModels: [],
    });

    expect(resolved).toMatchObject({ matchedModelName: 'My Model', status: 'installed' });
  });

  it('resolves a slot requirement as installed when any installed model matches the base (+ type)', () => {
    const requirement: WorkflowModelRequirement = {
      base: 'sdxl',
      kind: 'slot',
      label: 'SDXL checkpoint',
      modelType: 'main',
    };
    const installedModels: InstalledModelSummary[] = [
      { base: 'sdxl', hash: 'h', key: 'k', name: 'Some SDXL Checkpoint', type: 'main' },
    ];

    const [resolved] = resolveWorkflowModelRequirements([requirement], {
      activeInstallSources: new Set(),
      installedModels,
      starterModels: [],
    });

    expect(resolved).toMatchObject({ matchedModelName: 'Some SDXL Checkpoint', status: 'installed' });
  });

  it('matches a starter catalog entry via previous_names, case-insensitively', () => {
    const requirement: WorkflowModelRequirement = {
      identifier: { base: 'sdxl', key: 'missing-key', name: 'old name', type: 'main' },
      kind: 'exact',
      label: 'old name',
    };
    const starterModels: StarterCatalogEntry[] = [
      {
        base: 'sdxl',
        is_installed: false,
        name: 'New Name',
        previous_names: ['Old Name'],
        source: 'hf/some-model',
        type: 'main',
      },
    ];

    const [resolved] = resolveWorkflowModelRequirements([requirement], {
      activeInstallSources: new Set(),
      installedModels: [],
      starterModels,
    });

    expect(resolved?.status).toBe('installable');
    expect(resolved?.starterMatch?.name).toBe('New Name');
  });

  it('is installing when the matched starter source is in activeInstallSources', () => {
    const requirement: WorkflowModelRequirement = {
      base: 'sdxl',
      kind: 'slot',
      label: 'SDXL checkpoint',
      modelType: 'main',
    };
    const starterModels: StarterCatalogEntry[] = [
      { base: 'sdxl', is_installed: false, name: 'Some SDXL Checkpoint', source: 'hf/some-model', type: 'main' },
    ];

    const [resolved] = resolveWorkflowModelRequirements([requirement], {
      activeInstallSources: new Set(['hf/some-model']),
      installedModels: [],
      starterModels,
    });

    expect(resolved?.status).toBe('installing');
  });

  it('is installing when a dependency source of the matched starter is in activeInstallSources', () => {
    const requirement: WorkflowModelRequirement = {
      base: 'sdxl',
      kind: 'slot',
      label: 'SDXL checkpoint',
      modelType: 'main',
    };
    const starterModels: StarterCatalogEntry[] = [
      {
        base: 'sdxl',
        dependencies: [{ is_installed: false, source: 'hf/dependency' }],
        is_installed: false,
        name: 'Some SDXL Checkpoint',
        source: 'hf/some-model',
        type: 'main',
      },
    ];

    const [resolved] = resolveWorkflowModelRequirements([requirement], {
      activeInstallSources: new Set(['hf/dependency']),
      installedModels: [],
      starterModels,
    });

    expect(resolved?.status).toBe('installing');
  });

  it('is unresolvable when nothing installed or in the starter catalog matches', () => {
    const requirement: WorkflowModelRequirement = {
      base: 'flux',
      kind: 'slot',
      label: 'FLUX checkpoint',
      modelType: 'main',
    };

    const [resolved] = resolveWorkflowModelRequirements([requirement], {
      activeInstallSources: new Set(),
      installedModels: [],
      starterModels: [],
    });

    expect(resolved).toMatchObject({ matchedModelName: null, starterMatch: null, status: 'unresolvable' });
  });
});

describe('getAddModelsSearchTerm', () => {
  const resolve = (
    requirement: WorkflowModelRequirement,
    status: ResolvedModelRequirement['status'],
    starterMatch: StarterCatalogEntry | null = null
  ): ResolvedModelRequirement => ({ matchedModelName: null, requirement, starterMatch, status });

  const starter = (name: string): StarterCatalogEntry => ({
    base: 'wan',
    is_installed: false,
    name,
    source: 'https://models.test/starter',
    type: 'main',
  });

  const fluxSlot: WorkflowModelRequirement = {
    base: 'flux',
    kind: 'slot',
    label: 'FLUX checkpoint',
    modelType: 'main',
  };

  it('sends an installable row to the catalog entry that can supply it, by name', () => {
    expect(getAddModelsSearchTerm(resolve(fluxSlot, 'installable', starter('FLUX.1 dev')))).toBe('FLUX.1 dev');
  });

  it('has nothing to offer for a requirement that is already installed', () => {
    // Add Models searches the starter catalog, which need not carry an
    // installed model's name at all.
    expect(getAddModelsSearchTerm(resolve(fluxSlot, 'installed', starter('FLUX.1 dev')))).toBeNull();
  });

  it("falls back to the exact requirement's model name when the catalog has no match", () => {
    expect(
      getAddModelsSearchTerm(
        resolve(
          {
            identifier: { base: 'sdxl', key: 'model-key', name: 'Juggernaut XL' },
            kind: 'exact',
            label: 'Juggernaut XL',
          },
          'unresolvable'
        )
      )
    ).toBe('Juggernaut XL');
  });

  it("falls back to an exact requirement's base when the graph recorded no name", () => {
    expect(
      getAddModelsSearchTerm(
        resolve({ identifier: { base: 'sdxl', key: 'model-key' }, kind: 'exact', label: 'model-key' }, 'unresolvable')
      )
    ).toBe('sdxl');
  });

  it("falls back to a slot's raw base, not its prose label", () => {
    // "FLUX checkpoint" is a label; the catalog's index carries 'flux'.
    expect(getAddModelsSearchTerm(resolve(fluxSlot, 'unresolvable'))).toBe('flux');
  });

  it("falls back to a base-less slot's model type", () => {
    expect(
      getAddModelsSearchTerm(resolve({ base: null, kind: 'slot', label: 'VAE', modelType: 'vae' }, 'unresolvable'))
    ).toBe('vae');
  });

  it('has nothing to search for a requirement that describes neither', () => {
    expect(
      getAddModelsSearchTerm(resolve({ base: null, kind: 'slot', label: 'Model', modelType: null }, 'unresolvable'))
    ).toBeNull();
    expect(
      getAddModelsSearchTerm(
        resolve({ identifier: { key: 'model-key' }, kind: 'exact', label: 'model-key' }, 'unresolvable')
      )
    ).toBeNull();
  });
});
