/** Cover every upscale builder against the backend-validated node/field/literal/connection fixture. */

import type { BackendGraphContract } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import type { UpscaleWidgetValues } from './types';

import { compileUpscaleGraph } from './graph';
import { createDefaultUpscaleWidgetValues, getUpscaleValidationReasons, UPSCALE_ARCHITECTURES } from './settings';

seedArchitectureCapabilities();

const model = (key: string, type: string, base: string, extra: Record<string, unknown> = {}): ModelConfig =>
  ({
    base,
    file_size: 1,
    format: 'checkpoint',
    hash: `${key}-hash`,
    key,
    name: key,
    path: key,
    source: key,
    source_type: 'path',
    type,
    ...extra,
  }) as ModelConfig;

/** What the widget picks components from, the way it does for a user. */
const INSTALLED = [
  model('spandrel', 'spandrel_image_to_image', 'any'),
  model('t5', 't5_encoder', 'any'),
  model('clip', 'clip_embed', 'any', { variant: 'large' }),
  model('qwen3-4b', 'qwen3_encoder', 'any', { variant: 'qwen3_4b' }),
  model('qwen3-06b', 'qwen3_encoder', 'any', { variant: 'qwen3_06b' }),
  model('qwen35', 'qwen3_5_encoder', 'any', { variant: 'qwen3_5_4b' }),
  model('qwen3vl-4b', 'qwen3_vl_encoder', 'any', { variant: 'qwen3_vl_4b' }),
  model('flux-vae', 'vae', 'flux'),
  model('qwen-vae', 'vae', 'qwen-image'),
  model('sd1-tile', 'controlnet', 'sd-1', { name: 'sd1 tile' }),
  model('sdxl-tile', 'controlnet', 'sdxl', { name: 'sdxl tile' }),
  model('z-tile', 'controlnet', 'z-image', { name: 'Z-Image ControlNet Tile' }),
];

/**
 * One case per path a builder can take: CFG above 1 where the negative branch is CFG-gated, the
 * optional Z-Image ControlNet, a bundled Z-Image or Krea-2 install, and Anima-3.8B's second encoder.
 */
const CASES: { label: string; main: ModelConfig; patch?: Partial<UpscaleWidgetValues> }[] = [
  { label: 'sd-1', main: model('sd1', 'main', 'sd-1') },
  { label: 'sdxl', main: model('sdxl', 'main', 'sdxl') },
  { label: 'flux', main: model('flux', 'main', 'flux') },
  {
    label: 'z-image',
    main: model('z', 'main', 'z-image'),
    patch: {
      cfgScale: 4,
      tileControlnetModel: INSTALLED.find(({ key }) => key === 'z-tile') as UpscaleWidgetValues['tileControlnetModel'],
    },
  },
  { label: 'z-image-diffusers', main: model('z-diffusers', 'main', 'z-image', { format: 'diffusers' }) },
  { label: 'anima', main: model('anima', 'main', 'anima', { variant: 'anima_qwen3' }), patch: { cfgScale: 4.5 } },
  {
    label: 'anima-3.8b',
    main: model('anima-38', 'main', 'anima', { variant: 'anima_qwen35' }),
    patch: { cfgScale: 6 },
  },
  { label: 'krea-2-turbo', main: model('krea2-turbo', 'main', 'krea-2', { variant: 'krea2_turbo' }) },
  {
    label: 'krea-2-raw',
    main: model('krea2-raw', 'main', 'krea-2', { format: 'gguf_quantized', variant: 'krea2_base' }),
    patch: { cfgScale: 5.5 },
  },
  {
    label: 'krea-2-diffusers',
    main: model('krea2-diffusers', 'main', 'krea-2', { format: 'diffusers', variant: 'krea2_turbo' }),
  },
];

const compileCase = ({ main, patch }: (typeof CASES)[number]): BackendGraphContract => {
  const installed = [main, ...INSTALLED, model(`${main.key}-lora`, 'lora', main.base)];
  const defaults = createDefaultUpscaleWidgetValues(installed);
  const values: UpscaleWidgetValues = {
    ...defaults,
    inputImage: { height: 101, image_name: 'input.png', width: 203 },
    // Every builder's concept loader, so its node and edges reach the contract too.
    loras: [{ isEnabled: true, model: installed.at(-1) as UpscaleWidgetValues['loras'][number]['model'], weight: 1 }],
    ...patch,
  };

  expect(getUpscaleValidationReasons(values), `${main.key} is not satisfiable`).toEqual([]);

  return compileUpscaleGraph(values, 'gallery', { useCpuNoise: true }).backendGraph;
};

describe('upscale graph coverage', () => {
  it('covers every architecture the widget supports', () => {
    expect([...new Set(CASES.map(({ main }) => main.base))].sort()).toEqual(Object.keys(UPSCALE_ARCHITECTURES).sort());
  });

  it('emits the node types, fields and connections the backend has to provide', async () => {
    const byBase: Record<string, { nodeTypes: string[] }> = {};
    const connections = new Set<string>();
    const fields: Record<
      string,
      { inputs: Set<string>; literals: Map<string, Map<string, unknown>>; outputs: Set<string> }
    > = {};
    const fieldsFor = (nodeType: string) =>
      (fields[nodeType] ??= { inputs: new Set(), literals: new Map(), outputs: new Set() });

    for (const testCase of CASES) {
      const graph = compileCase(testCase);
      const entry = (byBase[testCase.main.base] ??= { nodeTypes: [] });

      entry.nodeTypes = [
        ...new Set([...entry.nodeTypes, ...Object.values(graph.nodes).map((node) => node.type)]),
      ].sort();

      for (const node of Object.values(graph.nodes)) {
        const { literals } = fieldsFor(node.type);

        for (const [field, value] of Object.entries(node)) {
          // `id` and `type` address the node; an undefined value is not serialized at all.
          if (field === 'id' || field === 'type' || value === undefined) {
            continue;
          }

          const values = literals.get(field) ?? new Map<string, unknown>();
          literals.set(field, values);

          // Scalars only: object values are model identifiers and images from this file's fixtures.
          if (value === null || typeof value !== 'object') {
            values.set(JSON.stringify(value), value);
          }
        }
      }

      for (const edge of graph.edges) {
        const source = graph.nodes[edge.source.node_id]!.type;
        const destination = graph.nodes[edge.destination.node_id]!.type;

        fieldsFor(source).outputs.add(edge.source.field);
        fieldsFor(destination).inputs.add(edge.destination.field);
        connections.add([source, edge.source.field, destination, edge.destination.field].join(' '));
      }
    }

    const contract = {
      connections: [...connections].sort(),
      _comment:
        'Generated by src/features/upscale/core/upscaleGraphCoverage.test.ts; regenerate with ' +
        '`vitest -u`. Excluded from oxfmt so the test alone owns its layout. Consumed by ' +
        'tests/app/invocations/test_frontend_graph_node_types.py, which checks every node type, ' +
        'edge field and literal input value below against the backend invocation registry, and ' +
        'every connection for whether the backend considers the two field types compatible. An ' +
        'empty literalInputs value list means the field is set to something other than a scalar, ' +
        'so only its name is checked.',
      byBase,
      fieldsByNodeType: Object.fromEntries(
        Object.entries(fields)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([nodeType, { inputs, literals, outputs }]) => [
            nodeType,
            {
              inputs: [...inputs].sort(),
              literalInputs: Object.fromEntries(
                [...literals]
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([field, values]) => [
                    field,
                    [...values].sort(([a], [b]) => a.localeCompare(b)).map(([, value]) => value),
                  ])
              ),
              outputs: [...outputs].sort(),
            },
          ])
      ),
    };

    await expect(`${JSON.stringify(contract, null, 2)}\n`).toMatchFileSnapshot(
      './__snapshots__/upscaleGraphNodeTypes.json'
    );
  });
});
