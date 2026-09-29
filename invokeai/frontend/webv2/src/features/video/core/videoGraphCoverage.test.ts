/**
 * Every supported video base, variant and generation mode compiles a graph — the systematic
 * counterpart to `graph.test.ts`, and the video twin of
 * `src/features/generation/core/graphCoverage.test.ts`.
 *
 * `graph.test.ts` asserts what an individual family wires, one hand-written case at a time. A
 * variant added to `VIDEO_GENERATION` but never given a case is silently untested, and so is a mode
 * a variant declares but no case exercises. This file iterates the registry itself, so a new
 * architecture is covered the moment it is registered.
 *
 * It also records the node types, edge fields and literal input values each case emits into a
 * committed fixture. That fixture is the frontend half of a cross-stack contract:
 * `tests/app/invocations/test_frontend_graph_node_types.py` reads it, alongside the Generate
 * panel's own, and checks every type and field against the backend's `InvocationRegistry`. The
 * frontend never sees a backend schema,
 * so a node renamed or a field moved is otherwise an enqueue-time error on a graph the user cannot
 * edit — after the model has loaded.
 *
 * The snapshot file is written by this test and by nothing else; `.oxfmtrc.json` excludes it so
 * the formatter and this test are not both in charge of the same bytes.
 */

import type { GenerationModelCatalogItem as ModelConfig, MainModelConfig } from '@features/generation/contracts';
import type { BackendGraphContract } from '@features/generation/core/contracts';

import { describe, expect, it } from 'vitest';

import type { VideoGenerationMode, VideoSettings, VideoSourceClip, VideoTargetResolution } from './types';
import type { VideoComponentPolicyContext, VideoComponentSlotPolicy, VideoComponentValueKey } from './videoPolicies';

import { compileVideoGraph } from './graph';
import {
  getDefaultVideoSettings,
  getVideoComponentSectionPolicy,
  getVideoValidationReasons,
  isAutoDurationActive,
  isTwoStageSupportedForMode,
  isVideoModelSelectable,
  SUPPORTED_VIDEO_BASES,
  VIDEO_GENERATION,
} from './videoPolicies';

const IMAGE = { height: 704, image_name: 'frame.png', width: 1248 };
const CLIP: VideoSourceClip = {
  endFrame: 47,
  fps: 24,
  height: 704,
  numFrames: 48,
  startFrame: 0,
  video_name: 'clip.mp4',
  width: 1248,
};

const { endFrame: _end, startFrame: _start, ...CLIP_REF } = CLIP;

/** The conditioning each mode needs; `resolveVideoMode` infers the mode back from it. */
const MODE_INPUTS: Record<VideoGenerationMode, Partial<VideoSettings>> = {
  'audio-to-video': { conditioningClip: { clip: CLIP_REF, fpsKnown: true, role: 'audio' } },
  extend: { sourceVideo: CLIP },
  'first-frame': { firstFrameImage: IMAGE },
  'first-last': { firstFrameImage: IMAGE, lastFrameImage: IMAGE },
  'last-frame': { lastFrameImage: IMAGE },
  reference: { references: [{ detail: 'max', image: IMAGE, kind: 'image' }] },
  txt2vid: {},
  'video-to-audio': { conditioningClip: { clip: CLIP_REF, fpsKnown: true, role: 'video' } },
};

/**
 * Candidate components for the slot filters to choose from — a search over a pool rather than a
 * hand-written model per slot, so the test agrees with the policy rather than with itself.
 */
const CANDIDATE_BASES = ['any', ...SUPPORTED_VIDEO_BASES] as const;
const CANDIDATE_VARIANTS = [undefined, 'fl2va', 'ref2va', 't2v_a14b', 'i2v_a14b', 'ti2v_5b', 'ltx2_dev'] as const;
const CANDIDATE_LATENT_CHANNELS = [undefined, 16, 48] as const;

const candidatesForSlot = (slot: VideoComponentSlotPolicy): ModelConfig[] => {
  const candidates: ModelConfig[] = [];

  for (const type of slot.modelTypes) {
    for (const base of CANDIDATE_BASES) {
      for (const variant of CANDIDATE_VARIANTS) {
        for (const latentChannels of type === 'vae' ? CANDIDATE_LATENT_CHANNELS : [undefined]) {
          candidates.push({
            base,
            // A component source has to be a bundled install; every other slot takes a
            // single file, and the H3/LTX-2 encoder slots do not care either way.
            format: slot.valueKind === 'main' ? 'diffusers' : 'checkpoint',
            key: `${base}-${type}-${variant ?? 'novariant'}${latentChannels ? `-${latentChannels}` : ''}`,
            name: `${base} ${type} ${variant ?? ''}`.trim(),
            type,
            variant: variant ?? null,
            ...(latentChannels ? { latent_channels: latentChannels } : {}),
          } as ModelConfig);
        }
      }
    }
  }

  return candidates;
};

const buildContext = (
  model: MainModelConfig,
  settings: VideoSettings,
  slots: readonly VideoComponentSlotPolicy[]
): VideoComponentPolicyContext => {
  const selectedComponents = {} as VideoComponentPolicyContext['selectedComponents'];

  for (const slot of slots) {
    selectedComponents[slot.key] = settings[slot.key] as never;
  }

  return { model, selectedComponents, settings };
};

/**
 * Fill every required slot with something its own filter accepts. Iterated to a fixpoint because
 * slots are interdependent: picking a component source can make another slot stop being required.
 */
const satisfyRequiredComponents = (
  model: MainModelConfig,
  initial: VideoSettings
): { settings: VideoSettings; filled: VideoComponentValueKey[] } => {
  let settings = initial;
  const filled: VideoComponentValueKey[] = [];

  for (let pass = 0; pass < 5; pass++) {
    const { slots } = getVideoComponentSectionPolicy(model, settings);
    const context = buildContext(model, settings, slots);
    let changed = false;

    for (const slot of slots) {
      if (!slot.required?.(context) || settings[slot.key]) {
        continue;
      }

      const candidate = candidatesForSlot(slot).find((c) => !slot.filter || slot.filter(c, context));

      if (candidate) {
        settings = { ...settings, [slot.key]: candidate as never };
        filled.push(slot.key);
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }

  return { filled: filled.sort(), settings };
};

interface Case {
  base: string;
  variant: string;
  mode: VideoGenerationMode;
  format: 'diffusers' | 'checkpoint';
  /** A preset to override the variant's default with; absent runs the default. */
  targetResolution?: VideoTargetResolution;
  /** Fill the optional LTX-2 duration head slot and let it choose the length. */
  autoDuration?: true;
  label: string;
}

const createModel = (testCase: Case): MainModelConfig =>
  ({
    base: testCase.base,
    format: testCase.format,
    key: `${testCase.base}-${testCase.variant}-${testCase.format}`,
    name: `${testCase.base} ${testCase.variant} (${testCase.format})`,
    type: 'main',
    variant: testCase.variant,
  }) as MainModelConfig;

/**
 * Both model shapes per variant: a bundled install supplies its own components, a single file
 * forces the component-source path, and the two produce different graphs. A shape the panel would
 * not let the user pick at all is dropped — which shapes those are is the policy's answer
 * (`isVideoModelSelectable`), not a list kept here: H3's Ref2VA folder install is supported state
 * but not runnable as the model, and hard-coding that would go stale.
 */
const cases: Case[] = SUPPORTED_VIDEO_BASES.flatMap((base) =>
  Object.entries(VIDEO_GENERATION[base].variants).flatMap(([variant, config]) => {
    // A multi-stage preset builds a different graph -- more denoise passes, and the nodes between
    // them -- so each one is its own case. Every one of them, not just the first: the fixture
    // records the literal values written into each field as well as the node types, and the widest
    // canvas is exactly the one a future bound on those fields would reject.
    const presets: (VideoTargetResolution | undefined)[] = [
      undefined,
      ...config.targetResolutions.filter((option) => option.stages !== undefined).map((option) => option.id),
    ];

    return config.modes.flatMap((mode) =>
      presets
        .filter((targetResolution) => {
          // A mode the policy refuses to run two-stage is not an uncovered case, it is a
          // combination that does not exist -- asked of the policy so the two cannot drift.
          const option = config.targetResolutions.find((entry) => entry.id === targetResolution);

          return option?.stages !== 2 || isTwoStageSupportedForMode(mode);
        })
        .flatMap((targetResolution) =>
          (['diffusers', 'checkpoint'] as const)
            .map((format) => ({
              base,
              format,
              label: `${base} / ${variant} / ${mode} / ${format}${targetResolution ? ` / ${targetResolution}` : ''}`,
              mode,
              ...(targetResolution ? { targetResolution } : {}),
              variant,
            }))
            .filter((testCase) => isVideoModelSelectable(createModel(testCase)))
        )
    );
  })
);

/**
 * Auto duration is an optional slot plus a flag, so the matrix above -- which fills only *required*
 * slots -- never reaches the `ltx2_duration` node. Added here rather than as another dimension of
 * that matrix because it only affects one base. Every preset is kept, including the two-stage ones:
 * those build a second denoise with a `num_frames` of its own, so restricting this to the default
 * preset is exactly how a graph that dies after the base pass stays invisible.
 */
const DURATION_HEAD = {
  base: 'ltx-2',
  key: 'ltx2-duration-head',
  name: 'LTX-2 duration head',
  type: 'ltx2_duration_head',
} as ModelConfig;

const AUTO_DURATION_CASES: Case[] = cases
  .filter((testCase) => testCase.base === 'ltx-2')
  .filter((testCase) =>
    // Asked of the policy, not of a list of modes kept here, so the two cannot drift.
    isAutoDurationActive({
      autoDuration: true,
      conditioningClip: null,
      firstFrameImage: null,
      lastFrameImage: null,
      ltx2DurationHeadModel: DURATION_HEAD,
      references: [],
      sourceVideo: null,
      ...MODE_INPUTS[testCase.mode],
    } as unknown as VideoSettings)
  )
  .map((testCase) => ({ ...testCase, autoDuration: true, label: `${testCase.label} / auto duration` }));

const compileForCase = (testCase: Case): { filled: VideoComponentValueKey[]; graph: BackendGraphContract } => {
  const model = createModel(testCase);
  const { filled, settings } = satisfyRequiredComponents(model, {
    ...getDefaultVideoSettings(model),
    ...MODE_INPUTS[testCase.mode],
    positivePrompt: 'a test prompt',
    seed: 1,
    seedMode: 'fixed',
    ...(testCase.targetResolution ? { targetResolution: testCase.targetResolution } : {}),
    ...(testCase.autoDuration ? { autoDuration: true, ltx2DurationHeadModel: DURATION_HEAD as never } : {}),
  });

  // Compiling an invalid selection throws only the first reason; asserting here reports every
  // unmet requirement at once, and doubles as the check that the policy is satisfiable at all.
  expect(getVideoValidationReasons(model, settings), `${testCase.label} is not satisfiable`).toEqual([]);

  return { filled, graph: compileVideoGraph(settings, model).backendGraph };
};

describe('video graph coverage', () => {
  it('registers a variant set for every supported base', () => {
    for (const base of SUPPORTED_VIDEO_BASES) {
      expect(Object.keys(VIDEO_GENERATION[base].variants).length, `${base} registers no variants`).toBeGreaterThan(0);
    }
  });

  it('compiles at least one case for every registered variant', () => {
    // The selectability filter above drops shapes the panel refuses; a variant it drops entirely
    // would leave this suite silently covering nothing for it.
    const covered = new Set(cases.map((testCase) => `${testCase.base}/${testCase.variant}`));
    const registered = SUPPORTED_VIDEO_BASES.flatMap((base) =>
      Object.keys(VIDEO_GENERATION[base].variants).map((variant) => `${base}/${variant}`)
    );

    expect([...covered].sort()).toEqual(registered.sort());
  });

  it.each(cases)('compiles a structurally sound graph for $label', (testCase) => {
    const { graph } = compileForCase(testCase);
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

  it('emits the node types and fields the backend has to provide', async () => {
    const byBase: Record<string, { modes: string[]; nodeTypes: string[] }> = {};
    const fields: Record<
      string,
      { inputs: Set<string>; literals: Map<string, Map<string, unknown>>; outputs: Set<string> }
    > = {};

    const connections = new Set<string>();
    const fieldsFor = (nodeType: string) =>
      (fields[nodeType] ??= { inputs: new Set(), literals: new Map(), outputs: new Set() });

    for (const testCase of [...cases, ...AUTO_DURATION_CASES]) {
      const { graph } = compileForCase(testCase);
      const entry = (byBase[testCase.base] ??= { modes: [], nodeTypes: [] });

      entry.modes = [...new Set([...entry.modes, testCase.mode])].sort();
      entry.nodeTypes = [
        ...new Set([...entry.nodeTypes, ...Object.values(graph.nodes).map((node) => node.type)]),
      ].sort();

      for (const node of Object.values(graph.nodes)) {
        const { literals } = fieldsFor(node.type);

        for (const [field, value] of Object.entries(node)) {
          // `id` and `type` address the node rather than feed it, and an undefined value is not
          // serialized into the request at all — the builders use it to mean "leave the default".
          if (field === 'id' || field === 'type' || value === undefined) {
            continue;
          }

          const values = literals.get(field) ?? new Map<string, unknown>();
          literals.set(field, values);

          // Scalars only: the object-valued inputs are model identifiers and media fields built
          // from this file's synthetic fixtures, so validating them would assert something about
          // the fixture rather than about the contract.
          if (value === null || typeof value !== 'object') {
            values.set(JSON.stringify(value), value);
          }
        }
      }

      for (const edge of graph.edges) {
        fieldsFor(graph.nodes[edge.source.node_id]!.type).outputs.add(edge.source.field);
        fieldsFor(graph.nodes[edge.destination.node_id]!.type).inputs.add(edge.destination.field);
        // The connection itself, by node TYPE, so the Python side can ask the backend whether the
        // two field types are actually compatible. Field names alone are not enough: a float wired
        // into an int is two valid names and a graph the queue refuses at enqueue.
        connections.add(
          [
            graph.nodes[edge.source.node_id]!.type,
            edge.source.field,
            graph.nodes[edge.destination.node_id]!.type,
            edge.destination.field,
          ].join(' ')
        );
      }
    }

    const contract = {
      connections: [...connections].sort(),
      _comment:
        'Generated by src/features/video/core/videoGraphCoverage.test.ts; regenerate with ' +
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
      './__snapshots__/videoGraphNodeTypes.json'
    );
  });
});
