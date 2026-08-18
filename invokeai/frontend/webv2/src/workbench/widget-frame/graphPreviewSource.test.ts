import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { InvocationTemplatesSnapshot } from '@features/workflow/react';
import type { GraphContract } from '@workbench/graphContracts';
import type { TFunction } from 'i18next';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { createGraphBearingSurface } from '@workbench/graphSurfaces';
import { canvasWidgetManifest } from '@workbench/widgets/canvas/manifest';
import { generateWidgetManifest } from '@workbench/widgets/generate/manifest';
import { workflowWidgetManifest } from '@workbench/widgets/workflow/manifest';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { describe, expect, it } from 'vitest';

import { buildGraphPreviewSource } from './graphPreviewSource';

const t = ((key: string) => key) as TFunction;

const sdxlModel = {
  base: 'sdxl',
  file_size: 0,
  format: 'diffusers',
  hash: 'sdxl-hash',
  key: 'sdxl-model',
  name: 'SDXL',
  path: 'sdxl.safetensors',
  source: 'sdxl.safetensors',
  source_type: 'path',
  type: 'main',
} satisfies ModelConfig;

const models: ModelConfig[] = [sdxlModel];

const createGenerateValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(sdxlModel),
  model: sdxlModel,
  modelKey: sdxlModel.key,
  positivePrompt: 'a sdxl prompt',
  ...overrides,
});

const getActiveProject = (values: GenerateWidgetValues) => {
  const state = workbenchReducer(createInitialWorkbenchState(), { type: 'setGenerateSettings', values });
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  expect(project).toBeDefined();

  return project!;
};

const idleTemplates: InvocationTemplatesSnapshot = { error: null, status: 'idle', templates: {} };
const loadedTemplates: InvocationTemplatesSnapshot = { error: null, status: 'loaded', templates: {} };

const project = getActiveProject(createGenerateValues());
const generateSurface = createGraphBearingSurface(generateWidgetManifest, 'left', 'Generate')!;
const workflowSurface = createGraphBearingSurface(workflowWidgetManifest, 'left', 'Workflow')!;
const canvasSurface = createGraphBearingSurface(canvasWidgetManifest, 'center', 'Canvas')!;

describe('buildGraphPreviewSource', () => {
  it('live-compiles the generate source and reports a seed notice when randomized', () => {
    const source = buildGraphPreviewSource({ models, project, surface: generateSurface, t, templates: idleTemplates });

    expect(source.isLive).toBe(true);
    expect(source.graph).not.toBeNull();
    expect(source.notices).toEqual([{ id: 'seed-random', message: 'graphPreview.seedRandomized', nodeId: 'seed' }]);
    expect(source.getProvenance?.('denoise_latents', 'steps')).toEqual({ label: 'graphPreview.provenance.steps' });
    expect(source.getProvenance?.('seed', 'value')).toEqual({ label: 'graphPreview.provenance.seedRandom' });
    expect(source.resolvedInputOverrides).toEqual({ seed: { value: 'graphPreview.seedRegenerated' } });
    expect(source.summaryRows.some((row) => row.id === 'steps')).toBe(true);
  });

  it('returns invalid reasons instead of a graph when generate settings cannot compile', () => {
    const invalidProject = getActiveProject(createGenerateValues({ width: 1000, height: 999 }));
    const source = buildGraphPreviewSource({
      models,
      project: invalidProject,
      surface: generateSurface,
      t,
      templates: idleTemplates,
    });

    expect(source.graph).toBeNull();
    expect(source.invalidReasons.length).toBeGreaterThan(0);
    expect(source.isLive).toBe(true);
  });

  it('compiles the workflow source from the live document with position hints', () => {
    const source = buildGraphPreviewSource({
      models: undefined,
      project,
      surface: workflowSurface,
      t,
      templates: loadedTemplates,
    });

    expect(source.isLive).toBe(true);
    expect(source.positionHints).toBeDefined();
  });

  it('returns null graph for workflow when templates are not loaded', () => {
    const source = buildGraphPreviewSource({
      models: undefined,
      project,
      surface: workflowSurface,
      t,
      templates: idleTemplates,
    });

    expect(source.graph).toBeNull();
    expect(source.isLive).toBe(true);
  });

  it('falls back to the last compiled widget graph for canvas', () => {
    const canvasGraph: GraphContract = {
      edges: [],
      id: 'canvas-graph',
      label: 'Canvas',
      nodes: [],
      updatedAt: '',
      version: 1,
    };
    const canvasProject = { ...project, widgetGraphs: { ...project.widgetGraphs, canvas: canvasGraph } };
    const source = buildGraphPreviewSource({
      models: undefined,
      project: canvasProject,
      surface: canvasSurface,
      t,
      templates: idleTemplates,
    });

    expect(source.isLive).toBe(false);
    expect(source.graph).toBe(canvasGraph);
    expect(source.notices).toEqual([]);
    expect(source.summaryRows).toEqual([]);
  });
});
