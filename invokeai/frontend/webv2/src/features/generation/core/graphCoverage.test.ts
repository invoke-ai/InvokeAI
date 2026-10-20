/** Cover every builder against backend-validated node/field/literal fixtures; tests own snapshot formatting. */

import type { BackendGraphContract } from '@features/generation/core/contracts';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import type { GenerateComponentValueKey, SupportedGenerateBase } from './baseGenerationPolicies';
import type { MainModelConfig, ModelIdentifierConfig, VaeModelConfig } from './types';

import {
  getComponentSectionPolicy,
  getGenerationValidationReasons,
  isLoraSupported,
  SUPPORTED_GENERATE_BASES,
} from './baseGenerationPolicies';
import { compileGenerateGraph, GRAPH_BUILDERS } from './graph';
import {
  buildContext,
  candidatesForSlot,
  generateGraphCases,
  type ModelShape,
  satisfiedSettingsFor,
  SHAPE_OVERRIDES,
  tunedSettingsFor,
} from './graphCoverage.testing';
import { COMPEL_NODE_TYPES, supportsPromptAttentionSyntax } from './prompt/compel';
import { REFINER_DENOISE_NODE_ID } from './sdxlRefinerGraph';

const compileForShape = (
  base: SupportedGenerateBase,
  shape: ModelShape,
  tuned = false
): { filled: GenerateComponentValueKey[]; graph: BackendGraphContract } => {
  const { filled, model, projectSettings, settings } = tuned
    ? tunedSettingsFor(base, shape)
    : {
        ...satisfiedSettingsFor(base, shape),
        projectSettings: { useCpuNoise: true, useNsfwChecker: false, useWatermark: false },
      };

  // Assert every requirement before compilation for complete diagnostics.
  expect(getGenerationValidationReasons(model, settings), `${base}/${shape.label} is not satisfiable`).toEqual([]);

  return { filled, graph: compileGenerateGraph(settings, model, 'gallery', projectSettings).backendGraph };
};

const SDXL_REFINER_SHAPE: ModelShape = { label: 'diffusers-with-refiner', overrides: { format: 'diffusers' } };
const SDXL_REFINER_MODEL: MainModelConfig = {
  base: 'sdxl-refiner',
  key: 'sdxl-refiner',
  name: 'SDXL Refiner',
  type: 'main',
};

// The refiner is a per-image choice, not a component slot, so it needs its own SDXL case.
const refinerCase = () => {
  const { filled, model, settings } = satisfiedSettingsFor('sdxl', SDXL_REFINER_SHAPE);
  const withRefiner = { ...settings, refinerModel: SDXL_REFINER_MODEL };

  expect(getGenerationValidationReasons(model, withRefiner)).toEqual([]);

  return {
    base: 'sdxl' as const,
    filled,
    graph: compileGenerateGraph(withRefiner, model, 'gallery', {
      useCpuNoise: true,
      useNsfwChecker: false,
      useWatermark: false,
    }).backendGraph,
    shape: SDXL_REFINER_SHAPE,
  };
};

seedArchitectureCapabilities();

describe('generate graph coverage', () => {
  it('has a builder for every supported base and no builder for anything else', () => {
    expect(Object.keys(GRAPH_BUILDERS).sort()).toEqual([...SUPPORTED_GENERATE_BASES].sort());
  });

  it('declares shape overrides only for bases that exist', () => {
    expect(Object.keys(SHAPE_OVERRIDES).filter((base) => !SUPPORTED_GENERATE_BASES.includes(base as never))).toEqual(
      []
    );
  });

  // The prompt editor highlights and weights attention syntax by this capability; only Compel reads it.
  it.each(generateGraphCases)(
    'encodes $label prompts with Compel exactly when attention syntax is offered',
    ({ base, shape }) => {
      const { graph } = compileForShape(base, shape);
      const usesCompel = Object.values(graph.nodes).some((node) => COMPEL_NODE_TYPES.has(node.type));

      expect(usesCompel).toBe(supportsPromptAttentionSyntax(base));
    }
  );

  // SDXL attention syntax carries through the refiner stage, which re-encodes the prompts with its own encoders.
  it('conditions the SDXL refiner denoise with Compel encoders', () => {
    const { graph } = refinerCase();
    const conditioningSource = (field: string) => {
      const sources = graph.edges.filter(
        (edge) => edge.destination.node_id === REFINER_DENOISE_NODE_ID && edge.destination.field === field
      );

      expect(sources, `refiner denoise ${field} has no single source`).toHaveLength(1);

      return graph.nodes[sources[0]!.source.node_id]!.type;
    };

    for (const field of ['positive_conditioning', 'negative_conditioning']) {
      const sourceType = conditioningSource(field);

      expect(COMPEL_NODE_TYPES.has(sourceType), `refiner ${field} is encoded by ${sourceType}`).toBe(true);
    }
  });

  it.each(generateGraphCases)('compiles a structurally sound graph for $label', ({ base, shape }) => {
    const { graph } = compileForShape(base, shape);
    const nodeIds = new Set(Object.keys(graph.nodes));

    expect(nodeIds.size).toBeGreaterThan(0);

    for (const [id, node] of Object.entries(graph.nodes)) {
      expect(node.id, `node keyed '${id}' carries a mismatched id`).toBe(id);
      expect(node.type, `node '${id}' has no type`).toBeTruthy();
    }

    for (const edge of graph.edges) {
      const description = `${edge.source.node_id}.${edge.source.field} -> ${edge.destination.node_id}.${edge.destination.field}`;

      expect(nodeIds.has(edge.source.node_id), `dangling source in edge ${description}`).toBe(true);
      expect(nodeIds.has(edge.destination.node_id), `dangling destination in edge ${description}`).toBe(true);
      expect(edge.source.field, `edge ${description} has no source field`).toBeTruthy();
      expect(edge.destination.field, `edge ${description} has no destination field`).toBeTruthy();
    }
  });

  it('sends every VAE the picker offers into the graph wherever a VAE is required', () => {
    // Enumerate accepted VAE families at runtime after fixture seeding.
    const checked: string[] = [];

    for (const { base, shape } of generateGraphCases) {
      const { filled, model, settings } = satisfiedSettingsFor(base, shape);

      if (!filled.includes('vae')) {
        continue;
      }

      const { slots } = getComponentSectionPolicy(model, settings);
      const vaeSlot = slots.find((slot) => slot.key === 'vae')!;
      const context = buildContext(model, settings, slots);
      const offered = new Map<string, ModelIdentifierConfig>();

      for (const candidate of candidatesForSlot(vaeSlot)) {
        if (!vaeSlot.filter || vaeSlot.filter(candidate, context)) {
          offered.set(`${candidate.base}/${String(candidate.latent_channels)}`, candidate);
        }
      }

      for (const vae of offered.values()) {
        const { backendGraph } = compileGenerateGraph({ ...settings, vae: vae as VaeModelConfig }, model, 'gallery', {
          useCpuNoise: true,
          useNsfwChecker: false,
          useWatermark: false,
        });
        // Metadata records the selection whether or not the builder used it, so it does not count.
        const sent = Object.values(backendGraph.nodes).some(
          (node) => node.type !== 'core_metadata' && JSON.stringify(node).includes(`"${vae.key}"`)
        );

        expect(sent, `${base}/${shape.label} dropped the offered VAE ${vae.key}`).toBe(true);
        checked.push(`${base}/${shape.label}:${vae.base}/${String(vae.latent_channels)}`);
      }
    }

    // Require nonempty cross-base coverage after runtime seeding.
    expect(checked).toEqual(
      expect.arrayContaining([
        'anima/qwen3:qwen-image/undefined',
        'anima/qwen3:wan/16',
        'anima/qwen35:qwen-image/undefined',
        'krea-2/standalone-components:anima/undefined',
        'qwen-image/standalone-components:anima/undefined',
        'z-image/standalone-components:flux/undefined',
      ])
    );
  });

  // An image's metadata is what recall restores, so it must name exactly the LoRAs that reached inference.
  it.each(generateGraphCases)('$label loads concepts as policy says, and records only those', ({ base, shape }) => {
    const { model, settings } = satisfiedSettingsFor(base, shape);
    const lora = (key: string) => ({ base, key, name: key, type: 'lora' as const });
    const { backendGraph } = compileGenerateGraph(
      {
        ...settings,
        loras: [
          { isEnabled: true, model: lora('first'), weight: 0.4 },
          { isEnabled: false, model: lora('disabled'), weight: 1 },
          { isEnabled: true, model: lora('second'), weight: -0.8 },
        ],
      },
      model,
      'gallery',
      { useCpuNoise: true, useNsfwChecker: false, useWatermark: false }
    );
    const nodes = Object.values(backendGraph.nodes);
    const loaded = nodes
      .filter((node) => node.type === 'lora_selector')
      .map((node) => [(node.lora as { key: string }).key, node.weight]);
    const recorded = (nodes.find((node) => node.type === 'core_metadata')?.loras ?? []) as {
      model: { key: string };
      weight: number;
    }[];

    // The concept picker trusts this policy, so a builder that gains or loses a LoRA loader must update it.
    expect(loaded.length > 0, `${base}/${shape.label} LoRA support disagrees with isLoraSupported`).toBe(
      isLoraSupported(model)
    );
    expect(recorded.map((entry) => [entry.model.key, entry.weight])).toEqual(loaded);
  });

  it('emits the node types and fields the backend has to provide', async () => {
    const byBase: Record<string, { componentsFilled: Record<string, string[]>; nodeTypes: string[] }> = {};
    const fields: Record<
      string,
      { inputs: Set<string>; literals: Map<string, Map<string, unknown>>; outputs: Set<string> }
    > = {};

    const fieldsFor = (nodeType: string) =>
      (fields[nodeType] ??= { inputs: new Set(), literals: new Map(), outputs: new Set() });

    // Each shape compiles twice: at its defaults, and with every optional tuning and output filter switched on.
    const compilations = [
      ...generateGraphCases.flatMap(({ base, shape }) =>
        [false, true].map((tuned) => ({ base, shape, ...compileForShape(base, shape, tuned) }))
      ),
      refinerCase(),
    ];

    for (const { base, filled, graph, shape } of compilations) {
      const entry = (byBase[base] ??= { componentsFilled: {}, nodeTypes: [] });

      entry.componentsFilled[shape.label] = filled;
      entry.nodeTypes = [
        ...new Set([...entry.nodeTypes, ...Object.values(graph.nodes).map((node) => node.type)]),
      ].sort();

      for (const node of Object.values(graph.nodes)) {
        const { literals } = fieldsFor(node.type);

        for (const [field, value] of Object.entries(node)) {
          // Node identity and undefined values are excluded from serialized input contracts.
          if (field === 'id' || field === 'type' || value === undefined) {
            continue;
          }

          const values = literals.get(field) ?? new Map<string, unknown>();
          literals.set(field, values);

          // Check scalar values, not only object names, for synthetic unhashed model fixtures.
          if (value === null || typeof value !== 'object') {
            values.set(JSON.stringify(value), value);
          }
        }
      }

      for (const edge of graph.edges) {
        fieldsFor(graph.nodes[edge.source.node_id]!.type).outputs.add(edge.source.field);
        fieldsFor(graph.nodes[edge.destination.node_id]!.type).inputs.add(edge.destination.field);
      }
    }

    const contract = {
      _comment:
        'Generated by src/features/generation/core/graphCoverage.test.ts; regenerate with ' +
        '`vitest -u`. Excluded from oxfmt so the test alone owns its layout. Consumed by ' +
        'tests/app/invocations/test_frontend_graph_node_types.py, which checks every node type, ' +
        'edge field and literal input value below against the backend invocation registry. An ' +
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
      './__snapshots__/generateGraphNodeTypes.json'
    );
  });
});
