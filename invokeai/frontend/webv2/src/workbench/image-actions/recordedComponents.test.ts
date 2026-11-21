/**
 * Every component model a compiled graph hands to a loader reaches the metadata record under the key recall reads. A
 * node field that holds a model and appears neither in `RECORDED_COMPONENT_FIELDS` nor in the exclusions below fails
 * here, so a new loader cannot ship unrecorded.
 */

import type { BackendGraphContract } from '@features/generation/contracts';

import { RECORDED_COMPONENT_FIELDS } from '@features/generation/core/graphBuilder';
import {
  generateGraphCases,
  satisfiedSettingsFor,
  tunedSettingsFor,
} from '@features/generation/core/graphCoverage.testing';
import { compileGenerateGraph, getRecordedComponents } from '@features/generation/graph';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { compileUpscaleGraph } from '@features/upscale/core/graph';
import { CASES as UPSCALE_CASES, valuesForCase } from '@features/upscale/core/upscaleGraphCoverage.testing';
import { describe, expect, it } from 'vitest';

import {
  canvasRecordCases,
  canvasSettingsOf,
  compileCanvasRecordCase,
  compileGenerateRecordCase,
  generateRecordCases,
  PROJECT_SETTINGS,
} from './imageRecordCases.testing';
import { recordOf } from './recallContract.testing';

seedArchitectureCapabilities();

const REFERENCE = 'A reference model, which `ref_images` records with its image.';
const PROVIDER = 'The external provider model is the main model, recorded as `model`.';

/** Model-valued fields that are not components: `node type.field` and where the record holds them instead. */
const EXCLUDED: Readonly<Record<string, string>> = {
  'anima_lllite.control_model': 'The Anima inpaint adapter, which the Canvas records as `anima_inpaint_adapter`.',
  'anima_ip_adapter.ip_adapter_model': REFERENCE,
  'controlnet.control_model':
    'A Canvas control layer records its model with the layer; the Upscale tile ControlNet is an Upscale panel setting no record holds.',
  'flux_ip_adapter.ip_adapter_model': REFERENCE,
  'flux_redux.redux_model': REFERENCE,
  'gemini_image_generation.model': PROVIDER,
  'ip_adapter.ip_adapter_model': REFERENCE,
  'lora_selector.lora': 'Recorded in `loras`, read from the same nodes.',
  'openai_image_generation.model': PROVIDER,
  'seedream_image_generation.model': PROVIDER,
  'spandrel_image_to_image_autoscale.image_to_image_model': 'An Upscale panel setting, recorded as `upscale_model`.',
  't2i_adapter.t2i_adapter_model': 'A Canvas control layer records its model with the layer.',
  'z_image_control.control_model': 'The Upscale tile control is an Upscale panel setting no record holds.',
};

interface GraphCase {
  graph: BackendGraphContract;
  label: string;
  /** The case's own main model, the only `model` a loader may hold without recording it as a component. */
  mainKey: string;
  record: Record<string, unknown>;
}

const isModelIdentifier = (value: unknown): value is { key: string } =>
  typeof value === 'object' &&
  value !== null &&
  ['key', 'base', 'type'].every((field) => typeof (value as Record<string, unknown>)[field] === 'string');

const buildCases = (): GraphCase[] => {
  const cases: GraphCase[] = [];
  const add = (
    label: string,
    mainKey: string,
    compiled: Parameters<typeof recordOf>[0],
    batch: Parameters<typeof recordOf>[1]
  ) => cases.push({ graph: compiled.backendGraph, label, mainKey, record: recordOf(compiled, batch) });

  for (const testCase of generateRecordCases()) {
    add(`generate ${testCase.label}`, testCase.model.key, compileGenerateRecordCase(testCase), testCase.settings);
  }

  for (const { base, label, shape } of generateGraphCases) {
    const satisfied = satisfiedSettingsFor(base, shape);
    const tuned = tunedSettingsFor(base, shape);

    add(
      `generate satisfied ${label}`,
      satisfied.model.key,
      compileGenerateGraph(satisfied.settings, satisfied.model, 'gallery', PROJECT_SETTINGS),
      satisfied.settings
    );
    add(
      `generate tuned ${label}`,
      tuned.model.key,
      compileGenerateGraph(tuned.settings, tuned.model, 'gallery', tuned.projectSettings),
      tuned.settings
    );
  }

  for (const testCase of canvasRecordCases()) {
    add(`canvas ${testCase.label}`, testCase.model.key, compileCanvasRecordCase(testCase), canvasSettingsOf(testCase));
  }

  for (const testCase of UPSCALE_CASES) {
    const { values } = valuesForCase(testCase);

    add(
      `upscale ${testCase.label}`,
      values.model!.key,
      compileUpscaleGraph(values, 'gallery', { useCpuNoise: true }),
      values
    );
  }

  return cases;
};

describe('recorded components', () => {
  let cases: GraphCase[] | undefined;
  const getCases = () => (cases ??= buildCases());
  const loaderNodes = ({ graph }: GraphCase) =>
    Object.values(graph.nodes).filter(({ type }) => type !== 'core_metadata');

  it('lists every model a node field holds, or names why not', () => {
    const unlisted = new Set<string>();
    const used = new Set<string>();

    for (const testCase of getCases()) {
      for (const node of loaderNodes(testCase)) {
        for (const [field, value] of Object.entries(node)) {
          const slot = `${node.type}.${field}`;

          if (!isModelIdentifier(value) || RECORDED_COMPONENT_FIELDS[node.type]?.[field]) {
            continue;
          }

          if (slot in EXCLUDED) {
            used.add(slot);
          } else if (!(field === 'model' && node.type.endsWith('_model_loader') && value.key === testCase.mainKey)) {
            unlisted.add(slot);
          }
        }
      }
    }

    expect([...unlisted].sort(), 'model fields neither in RECORDED_COMPONENT_FIELDS nor excluded').toEqual([]);
    expect(
      Object.keys(EXCLUDED).filter((slot) => !used.has(slot)),
      'stale EXCLUDED entries: no case holds the field'
    ).toEqual([]);
  });

  it('records exactly the components the graph loads', () => {
    const componentKeys = new Set(
      Object.values(RECORDED_COMPONENT_FIELDS).flatMap((fields) => Object.values(fields).flat())
    );
    const failures: string[] = [];

    for (const { graph, label, record } of getCases()) {
      const derived = JSON.parse(JSON.stringify(getRecordedComponents(graph))) as Record<string, unknown>;

      for (const key of componentKeys) {
        if (JSON.stringify(record[key]) !== JSON.stringify(derived[key])) {
          failures.push(
            `${label}: \`${key}\` is ${JSON.stringify(record[key])}, the graph loads ${JSON.stringify(derived[key])}`
          );
        }
      }
    }

    expect(failures).toEqual([]);
  });

  it('has one value per key where several loaders hold a component', () => {
    const failures: string[] = [];

    for (const testCase of getCases()) {
      const seen = new Map<string, string>();

      for (const node of loaderNodes(testCase)) {
        for (const [field, keys] of Object.entries(RECORDED_COMPONENT_FIELDS[node.type] ?? {})) {
          const value = node[field];

          if (value === undefined || value === null) {
            continue;
          }

          for (const key of keys) {
            const previous = seen.get(key);

            if (previous !== undefined && previous !== JSON.stringify(value)) {
              failures.push(`${testCase.label}: \`${key}\` is written by two loaders, one is ${node.type}.${field}`);
            }

            seen.set(key, JSON.stringify(value));
          }
        }
      }
    }

    expect(failures).toEqual([]);
  });

  it('does not record a panel VAE the graph does not load', () => {
    const { model, settings } = satisfiedSettingsFor(
      'sdxl',
      generateGraphCases.find(({ base }) => base === 'sdxl')!.shape
    );
    const withVae = { ...settings, vae: { base: 'sd-1', key: 'sd-1-vae', name: 'SD 1.5 VAE', type: 'vae' as const } };
    const compiled = compileGenerateGraph(withVae, model, 'gallery', PROJECT_SETTINGS);

    expect(Object.values(compiled.backendGraph.nodes).some(({ type }) => type === 'vae_loader')).toBe(false);
    expect(recordOf(compiled, withVae)).not.toHaveProperty('vae');
  });
});
