import type { WorkingMemoryCase } from '@features/models';

import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import { toCalibrationGraphs, toGraphExportCase } from './calibrationGraphs';
import { exportGraphs } from './graphExport';

const MODELS = [
  {
    base: 'sdxl',
    format: 'diffusers',
    hash: 'blake3:sdxl',
    key: 'sdxl',
    name: 'SDXL',
    type: 'main',
    variant: 'normal',
  },
];

/** A case as the server's plan sends it for an SDXL decode in fp32. */
const planCase = (overrides: Partial<WorkingMemoryCase> = {}): WorkingMemoryCase => ({
  components: {},
  estimator: 'sdxl_vae_decode',
  height: 1088,
  id: 'sdxl_vae_decode-decode-fp32-1088x1088',
  line: 'decode_fp32',
  mode: 'txt2img',
  model_key: 'sdxl',
  negative_prompt: '',
  positive_prompt: 'a mountain lake at dawn, detailed photograph',
  seed: 1,
  settings: { vaePrecision: 'fp32' },
  steps: 2,
  width: 1088,
  ...overrides,
});

const compile = (cases: WorkingMemoryCase[]) =>
  toCalibrationGraphs(
    cases,
    exportGraphs({
      capabilities: architectureCapabilitiesFixture,
      cases: cases.map(toGraphExportCase),
      models: MODELS,
      version: 1,
    })
  );

const nodesOf = (graph: unknown) => Object.values((graph as { nodes: Record<string, Record<string, unknown>> }).nodes);

describe('calibration graphs', () => {
  it('compiles a plan case as Generate would, at its size, steps and decode precision', () => {
    const [compiled] = compile([planCase()]);
    const nodes = nodesOf(compiled!.graph);

    expect(compiled!.caseId).toBe('sdxl_vae_decode-decode-fp32-1088x1088');
    expect((compiled!.graph as { id: string }).id).toBe('sdxl_vae_decode-decode-fp32-1088x1088');
    expect(nodes.find((node) => node.type === 'noise')).toMatchObject({ height: 1088, width: 1088 });
    expect(nodes.find((node) => node.type === 'denoise_latents')).toMatchObject({ steps: 2 });
    expect(nodes.find((node) => node.type === 'l2i')).toMatchObject({ fp32: true });
    // Seed and prompt travel as batch data, as Generate queues them.
    expect(compiled!.data).toEqual([
      [
        { field_name: 'value', items: [1], node_path: 'seed' },
        { field_name: 'value', items: ['a mountain lake at dawn, detailed photograph'], node_path: 'positive_prompt' },
        { field_name: 'value', items: [''], node_path: 'negative_prompt' },
      ],
    ]);
  });

  it('keeps fp16 cases on the fp16 decode', () => {
    const [compiled] = compile([
      planCase({ id: 'fp16-case', line: 'decode_fp16', settings: { vaePrecision: 'fp16' } }),
    ]);
    const nodes = nodesOf(compiled!.graph);

    expect(nodes.find((node) => node.type === 'l2i')).toMatchObject({ fp32: false });
  });

  it('rejects naming the case that cannot compile, so nothing is queued', () => {
    expect(() => compile([planCase(), planCase({ id: 'missing-model', model_key: 'not-installed' })])).toThrow(
      /^Case missing-model: model_key "not-installed" is not a selectable main model/
    );
  });
});
