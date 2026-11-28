/**
 * What an empty component slot names as its supplier ("From <model>") must be what the compiled graph loads. Each
 * empty, optional slot the policy attributes is checked against the loader node of a graph compiled from the same
 * settings: a source is named in the loader's source field, the main model's own components leave the standalone
 * field empty. The expected values are read from the graph; only the slot-to-field table below is this test's own.
 */

import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  MainModelConfig,
} from '@features/generation/contracts';
import type {
  ComponentSlotPolicy,
  GenerateComponentValueKey,
  SupportedGenerateBase,
} from '@features/generation/core/baseGenerationPolicies';

import {
  getComponentPolicyContext,
  getComponentSectionPolicy,
  getComponentSlotsSuppliedByRecord,
  getEmptyComponentSupplier,
  getFlux2ComponentSource,
} from '@features/generation/core/baseGenerationPolicies';
import {
  candidatesForSlot,
  generateGraphCases,
  type ModelShape,
  satisfiedSettingsFor,
} from '@features/generation/core/graphCoverage.testing';
import { compileGenerateGraph } from '@features/generation/graph';
import { getGenerationUiPolicy } from '@features/generation/settings';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import { generateRecordCases, PROJECT_SETTINGS } from './imageRecordCases.testing';

seedArchitectureCapabilities();

/** The loader fields that hold a standalone component for each slot (a family names it differently). */
const STANDALONE_FIELDS: Partial<Record<GenerateComponentValueKey, readonly string[]>> = {
  clipEmbedModel: ['clip_embed_model'],
  clipGEmbedModel: ['clip_g_model'],
  clipLEmbedModel: ['clip_l_model'],
  ideogram4UnconditionalModel: ['unconditional_model'],
  mistralEncoderModel: ['mistral_encoder_model', 'text_encoder_model'],
  qwen35EncoderModel: ['qwen3_5_encoder_model'],
  qwen3EncoderModel: ['qwen3_encoder_model'],
  qwen3VLEncoderModel: ['qwen3_vl_encoder_model'],
  qwenVLEncoderModel: ['qwen_vl_encoder_model'],
  t5EncoderModel: ['t5_encoder_model'],
  vae: ['vae_model'],
  wanT5EncoderModel: ['wan_t5_encoder_model'],
};

/** The loader fields that name a component source. */
const SOURCE_FIELDS = ['component_source', 'mistral_source_model', 'qwen3_source_model'] as const;

/**
 * What each loader does when both its main and a component source could supply a component, read from the Python
 * loaders in invokeai/app/invocations (standalone model first in all of them):
 * - `qwen_image/qwen_image_model_loader.py`, `wan/wan_model_loader.py`, `flux2/flux2_dev_model_loader.py`,
 *   `flux2/flux2_klein_model_loader.py`: the main when it bundles its components, else the source.
 * - `z_image/z_image_model_loader.py`: the source, else the main when it is a self-contained pipeline.
 */
const MAIN_FIRST_LOADERS = new Set([
  'flux2_dev_model_loader',
  'flux2_klein_model_loader',
  'qwen_image_model_loader',
  'wan_model_loader',
]);

/** Whether the loader reads submodels from this main: a Diffusers folder, or a self-contained SDNQ pipeline. */
const mainBundles = (model: MainModelConfig): boolean =>
  model.format === 'diffusers' ||
  (model.format === 'sdnq_quantized' &&
    ['text_encoder', 'tokenizer', 'transformer', 'vae'].every((name) =>
      Boolean((model.submodels as Record<string, unknown> | undefined)?.[name])
    ));

/** Optional slots the policy leaves unattributed, and why their empty value is no model's component. */
const UNATTRIBUTED: Readonly<Partial<Record<GenerateComponentValueKey, string>>> = {
  animaInpaintAdapterModel: 'empty means no adapter',
  componentSourceModel: 'the source itself',
  flux3PromptUpsamplerModel: 'empty means no upsampler',
  gemma2EncoderModel: 'a PiD component: empty while PiD is off, no main model holds it',
  pidDecoderModel: 'a PiD component: empty while PiD is off, no main model holds it',
  qwen3VLVisionTowerModel: 'empty means no vision tower; a bundled encoder reads references itself',
  wanLowNoiseModel:
    'a Diffusers A14B main supplies its own second expert (Transformer2), a single-file or TI2V-5B main has none: ' +
    'empty is not one model for every main',
};

interface Case {
  label: string;
  model: MainModelConfig;
  settings: GenerateSettings;
}

const isMain = (model: GenerateModelConfig): model is MainModelConfig => model.type === 'main';

const UNLISTED_SOURCE_SLOT: ComponentSlotPolicy = {
  key: 'componentSourceModel',
  label: 'Component source',
  modelTypes: ['main'],
  valueKind: 'main',
};

const SDNQ_PIPELINE = { text_encoder: {}, tokenizer: {}, transformer: {}, vae: {} };

/** Self-contained SDNQ pipelines, which bundle what a Diffusers install does (FLUX.1 also its second encoder). */
const SDNQ_SHAPES: ReadonlyArray<{ base: SupportedGenerateBase; shape: ModelShape }> = [
  {
    base: 'flux',
    shape: {
      label: 'sdnq-pipeline',
      overrides: {
        format: 'sdnq_quantized',
        submodels: { ...SDNQ_PIPELINE, text_encoder_2: {}, tokenizer_2: {} },
      } as ModelShape['overrides'],
    },
  },
  {
    base: 'flux2',
    shape: {
      label: 'klein-sdnq-pipeline',
      overrides: { format: 'sdnq_quantized', submodels: SDNQ_PIPELINE, variant: 'klein_9b' } as ModelShape['overrides'],
    },
  },
  {
    base: 'z-image',
    shape: {
      label: 'sdnq-pipeline',
      overrides: { format: 'sdnq_quantized', submodels: SDNQ_PIPELINE } as ModelShape['overrides'],
    },
  },
];

/** The satisfied shapes with nothing optional filled, then the same with a component source in place of their own. */
const shapeCases = (): Case[] =>
  [...generateGraphCases, ...SDNQ_SHAPES].flatMap(({ base, shape }): Case[] => {
    const { model, settings } = satisfiedSettingsFor(base as SupportedGenerateBase, shape);
    const label = `${base} / ${shape.label}`;
    const { slots } = getComponentSectionPolicy(model, settings);
    const sourceSlot = slots.find((slot) => slot.key === 'componentSourceModel') ?? UNLISTED_SOURCE_SLOT;
    const context = getComponentPolicyContext(model, settings);
    const source = candidatesForSlot(sourceSlot).find(
      (candidate) =>
        (sourceSlot === UNLISTED_SOURCE_SLOT
          ? getFlux2ComponentSource(model, { ...settings, componentSourceModel: candidate as MainModelConfig })
          : sourceSlot.filter?.(candidate, context)) && candidate.key !== model.key
    ) as MainModelConfig | undefined;
    const cases: Case[] = [{ label, model, settings: { ...settings, qwen3VLVisionTowerModel: null } }];

    if (source) {
      // What the source supplies is optional now: clear the slots no longer required.
      let sourced: GenerateSettings = { ...settings, componentSourceModel: source };

      for (const slot of getComponentSectionPolicy(model, sourced).slots) {
        if (slot.key !== 'componentSourceModel' && !slot.required?.(getComponentPolicyContext(model, sourced))) {
          sourced = { ...sourced, [slot.key]: null };
        }
      }

      cases.push({ label: `${label} (component source)`, model, settings: sourced });
    }

    return cases;
  });

/** The loader a graph compiled for the case holds, or null where validation refuses the settings. */
const loaderOf = ({ model, settings }: Case) => {
  try {
    const { backendGraph } = compileGenerateGraph(settings, model, 'gallery', PROJECT_SETTINGS);

    return {
      graph: backendGraph,
      loader: backendGraph.nodes.model_loader as Record<string, unknown> | undefined,
    };
  } catch {
    return null;
  }
};

const keyOf = (value: unknown): string | undefined =>
  value && typeof value === 'object' ? ((value as { key?: string }).key ?? undefined) : undefined;

describe('empty component supplier', () => {
  it('names the model the compiled loader takes each empty optional component from', () => {
    const cases = [
      ...shapeCases(),
      ...generateRecordCases()
        .filter((testCase): testCase is typeof testCase & { model: MainModelConfig } => isMain(testCase.model))
        .map(({ label, model, settings }) => ({ label, model, settings })),
    ];
    const failures: string[] = [];
    const checked = { main: new Set<string>(), mainOverSource: new Set<string>(), source: new Set<string>() };
    const unattributed = new Set<GenerateComponentValueKey>();

    for (const testCase of cases) {
      const { label, model, settings } = testCase;
      const compiled = loaderOf(testCase);

      if (!compiled) {
        // Only a component-source case may be refused; a refused shape would hide its slots from this check.
        if (!label.endsWith('(component source)')) {
          failures.push(`\`${label}\` does not compile`);
        }

        continue;
      }

      const context = getComponentPolicyContext(model, settings);
      const slots = getComponentSectionPolicy(model, settings).slots.map((slot) => slot.key);
      // The SD VAE override has no slot in the policy.
      const policySlots = getComponentSectionPolicy(model, settings).slots;
      const hasSdVae = getGenerationUiPolicy(model).sdVaeVisible;
      const keys = new Set<GenerateComponentValueKey>([...slots, ...(hasSdVae ? ['vae' as const] : [])]);

      for (const key of keys) {
        const slot = policySlots.find((candidate) => candidate.key === key);

        if (settings[key] || slot?.required?.(context)) {
          continue;
        }

        const supplier = getEmptyComponentSupplier(model, settings, key);

        if (!supplier) {
          unattributed.add(key);
          continue;
        }

        const where = `\`${label}\` slot \`${key}\` names ${supplier.key}`;
        const { graph, loader } = compiled;

        if (hasSdVae) {
          // The SD graph loads an override through its own VAE node; without one the main model's VAE decodes.
          if (Object.values(graph.nodes).some((node) => node.type === 'vae_loader')) {
            failures.push(`${where}, but the graph loads a standalone VAE`);
          }

          checked.main.add(model.base);
          continue;
        }

        if (!loader) {
          failures.push(`${where}, but the graph has no \`model_loader\``);
          continue;
        }

        const standalone = (STANDALONE_FIELDS[key] ?? []).filter((field) => loader[field] !== undefined);
        const sources = SOURCE_FIELDS.flatMap((field) => (loader[field] === undefined ? [] : [keyOf(loader[field])]));

        if (!STANDALONE_FIELDS[key]) {
          failures.push(`${where}: this test has no loader field for the slot`);
        } else if (standalone.length > 0) {
          failures.push(`${where}, but the loader holds a standalone model in ${standalone.join(', ')}`);
        } else {
          // Who the loader takes it from, by its own precedence: no source field means the main, as for every family
          // without a component source.
          const sourceKey = sources[0];
          const loaded =
            sourceKey === undefined
              ? model.key
              : MAIN_FIRST_LOADERS.has(String(loader.type))
                ? mainBundles(model)
                  ? model.key
                  : sourceKey
                : sourceKey;

          if (loaded !== supplier.key) {
            failures.push(`${where}, but the loader takes it from ${loaded}`);
          } else if (loaded === model.key) {
            checked.main.add(model.base);
          } else {
            checked.source.add(model.base);
          }

          // A source in the panel never reaches the loader of a Diffusers main, whichever order the loader prefers.
          if (
            MAIN_FIRST_LOADERS.has(String(loader.type)) &&
            model.format === 'diffusers' &&
            settings.componentSourceModel &&
            sources.length === 0
          ) {
            checked.mainOverSource.add(model.base);
          }
        }
      }
    }

    expect(failures).toEqual([]);
    // Each family that takes components from its main, and each that takes them from a source, was exercised.
    expect([...checked.main].sort()).toEqual(
      [
        'ernie-image',
        'flux',
        'flux2',
        'flux3',
        'ideogram-4',
        'krea-2',
        'moss-image',
        'qwen-image',
        'qwen-image-2-1',
        'sd-1',
        'sd-2',
        'sd-3',
        'sdxl',
        'wan',
        'z-image',
      ].sort()
    );
    // A Diffusers main supplies its components while a source is set in the panel, which the loader never receives.
    expect([...checked.mainOverSource].sort()).toEqual(['flux2', 'qwen-image', 'wan']);
    expect([...checked.source].sort()).toEqual(['flux2', 'qwen-image', 'wan', 'z-image']);
    expect([...unattributed].filter((key) => !UNATTRIBUTED[key])).toEqual([]);
    expect(Object.keys(UNATTRIBUTED).filter((key) => !unattributed.has(key as GenerateComponentValueKey))).toEqual([]);
  });

  it('attributes nothing while another model, or none, would be loaded', () => {
    const { model, settings } = satisfiedSettingsFor(
      'wan',
      generateGraphCases.find(({ base }) => base === 'wan')!.shape
    );

    for (const key of Object.keys(UNATTRIBUTED) as GenerateComponentValueKey[]) {
      expect(getEmptyComponentSupplier(model, settings, key), key).toBeNull();
    }

    expect(getEmptyComponentSupplier(undefined, settings, 'vae')).toBeNull();
    expect(
      getEmptyComponentSupplier(
        { ...model, format: 'gguf_quantized' } as ComponentModelConfig as MainModelConfig,
        {},
        'vae'
      )
    ).toBeNull();
  });

  // The loader rejects a Diffusers source whose VAE is not the width the main decodes with: TI2V-5B decodes with the
  // 48-channel VAE, the A14B variants with the 16-channel one. The same source can still supply the text encoder.
  it.each([
    ['ti2v_5b', 't2v_a14b', null],
    ['t2v_a14b', 'ti2v_5b', null],
    ['t2v_a14b', 'i2v_a14b', 'wan-source'],
    ['ti2v_5b', 'ti2v_5b', 'wan-source'],
  ] as const)(
    'names a %s Wan main with a %s source as the VAE supplier: %s',
    (mainVariant, sourceVariant, supplierKey) => {
      const wan = (format: string, key: string, variant: string) =>
        ({ base: 'wan', format, key, name: key, type: 'main', variant }) as unknown as MainModelConfig;
      const main = wan('gguf_quantized', 'wan-main', mainVariant);
      const source = wan('diffusers', 'wan-source', sourceVariant);

      expect(getEmptyComponentSupplier(main, { componentSourceModel: source }, 'vae')?.key ?? null).toBe(supplierKey);
      expect(getEmptyComponentSupplier(main, { componentSourceModel: source }, 'wanT5EncoderModel')?.key).toBe(
        source.key
      );
    }
  );

  it.each([
    ['gguf_quantized', null],
    ['diffusers', 'sd3-main'],
  ] as const)('names an SD 3 %s main as the supplier only when it is a Diffusers pipeline', (format, supplierKey) => {
    const model = { base: 'sd-3', format, key: 'sd3-main', name: 'SD 3.5', type: 'main' } as MainModelConfig;

    for (const key of ['clipGEmbedModel', 'clipLEmbedModel', 't5EncoderModel', 'vae'] as const) {
      expect(getEmptyComponentSupplier(model, {}, key)?.key ?? null, key).toBe(supplierKey);
    }
  });

  // `source`: none recorded, a Diffusers pipeline of the model's own kind, or one recall cannot resolve.
  it.each([
    ['z-image', { format: 'diffusers' }, 'none', ['qwen3EncoderModel', 'vae']],
    // The Z-Image loader reads the source first, so an unknown one may have supplied everything.
    ['z-image', { format: 'diffusers' }, 'unresolved', []],
    ['z-image', { format: 'gguf_quantized' }, 'none', []],
    ['z-image', { format: 'gguf_quantized' }, 'source', ['qwen3EncoderModel', 'vae']],
    ['qwen-image', { format: 'diffusers' }, 'none', ['qwenVLEncoderModel', 'vae']],
    // A bundled main fills its slots ahead of any source, known or not.
    ['qwen-image', { format: 'diffusers' }, 'source', ['qwenVLEncoderModel', 'vae']],
    ['qwen-image', { format: 'diffusers' }, 'unresolved', ['qwenVLEncoderModel', 'vae']],
    ['qwen-image', { format: 'gguf_quantized' }, 'unresolved', []],
    ['wan', { format: 'diffusers' }, 'source', ['vae', 'wanT5EncoderModel']],
    ['flux2', { format: 'diffusers', variant: 'klein_4b' }, 'none', ['qwen3EncoderModel', 'vae']],
    ['flux2', { format: 'diffusers', variant: 'dev' }, 'source', ['mistralEncoderModel', 'vae']],
    ['flux2', { format: 'gguf_quantized', variant: 'dev' }, 'none', []],
    ['flux2', { format: 'gguf_quantized', variant: 'dev' }, 'source', ['mistralEncoderModel', 'vae']],
    ['krea-2', { format: 'diffusers' }, 'none', ['qwen3VLEncoderModel', 'vae']],
    ['sdxl', { format: 'diffusers' }, 'none', ['vae']],
    ['sd-1', { format: 'checkpoint' }, 'none', ['vae']],
  ] as const)('lists the slots a %s %j record with %s source left to its loader', (base, overrides, kind, expected) => {
    const { model } = satisfiedSettingsFor(base, { label: 'main', overrides });
    const source =
      kind === 'none'
        ? null
        : kind === 'unresolved'
          ? kind
          : ({ ...model, format: 'diffusers', key: `${base}-source`, submodels: undefined } as MainModelConfig);

    expect(getComponentSlotsSuppliedByRecord(model, source, {}).sort()).toEqual(expected);
  });
});
