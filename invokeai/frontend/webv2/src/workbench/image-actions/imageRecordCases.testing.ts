/**
 * The compiled image cases the record gates share, as definitions: model, settings and, for the Canvas, mode and
 * compile extras. A consumer that varies a setting recompiles through the compile helpers, which hold what belongs to
 * the case rather than to its settings. Call them where the architecture table is seeded.
 */

import type { ComponentModelConfig, GenerateModelConfig, GenerateSettings } from '@features/generation/contracts';
import type { CanvasCompileMode } from '@features/generation/core/canvas/types';

import { canvasModesFor, compile as compileCanvas } from '@features/generation/core/canvas/compileCanvasGraph.testing';
import {
  externalRecallCases,
  flux2DevSourceRecallCase,
  flux3UpsamplerRecallCase,
  fluxReduxRecallCase,
  generateGraphCases,
  krea2EditRecallCase,
  recallTunedSettingsFor,
  sdxlRefinerRecallCase,
} from '@features/generation/core/graphCoverage.testing';
import { compileGenerateGraph } from '@features/generation/graph';

export const PROJECT_SETTINGS = { useCpuNoise: true, useNsfwChecker: false, useWatermark: false };

export interface GenerateRecordCase {
  label: string;
  model: GenerateModelConfig;
  /** Installed models the case's panel lists beside its own, ahead of them. */
  otherModels?: readonly ComponentModelConfig[];
  settings: GenerateSettings;
}

type CanvasCompileExtras = Omit<NonNullable<Parameters<typeof compileCanvas>[2]>, 'bbox' | 'settings'>;

export interface CanvasRecordCase {
  /** The compile extras the case adds: control layers, regions, Canvas metadata. */
  compile?: CanvasCompileExtras;
  /** The frame the bbox covers, where it is not the settings' size. */
  frame?: { height: number; width: number };
  isCanvasFrame?: true;
  label: string;
  mode: CanvasCompileMode;
  model: GenerateModelConfig;
  /** Panel settings the recall baseline holds off the default, so a record turning one on shows. */
  panelOverrides?: Partial<GenerateSettings>;
  settings: GenerateSettings;
}

const ANIMA_INPAINT_ADAPTER: ComponentModelConfig = {
  base: 'anima',
  cond_in_channels: 4,
  key: 'anima-inpaint-adapter',
  name: 'Anima LLLite Inpainting',
  type: 'controlnet',
};

const CANVAS_REGION = {
  autoNegative: false,
  id: 'region',
  maskBounds: { height: 256, width: 192, x: 64, y: 128 },
  maskImageName: 'region-mask.png',
  negativePrompt: null,
  positivePrompt: 'a cat',
  referenceImages: [],
};

/** Every Generate shape tuned off its recall defaults, then the cases a shape alone does not reach. */
export const generateRecordCases = (): GenerateRecordCase[] => [
  ...generateGraphCases.map(({ base, shape }) => recallTunedSettingsFor(base, shape)),
  sdxlRefinerRecallCase(),
  flux2DevSourceRecallCase(),
  flux3UpsamplerRecallCase(),
  fluxReduxRecallCase(),
  krea2EditRecallCase(),
  ...externalRecallCases(),
];

/** One shape per base (the Canvas adds the same keys to every shape's record) in every mode it offers, then extras. */
export const canvasRecordCases = (): CanvasRecordCase[] => {
  const tunedShapes = generateGraphCases.map(({ base, shape }) => recallTunedSettingsFor(base, shape));
  const firstShapes = tunedShapes.filter(
    (fixture, index) => tunedShapes.findIndex((other) => other.model.base === fixture.model.base) === index
  );
  const fixtureFor = (base: string) => firstShapes.find((fixture) => fixture.model.base === base)!;
  const anima = fixtureFor('anima');
  const sdxl = fixtureFor('sdxl');
  const flux3 = fixtureFor('flux3');
  const [, gemini] = externalRecallCases();

  return [
    ...firstShapes.flatMap((fixture) =>
      canvasModesFor(fixture.model, fixture.settings).map((mode): CanvasRecordCase => ({
        ...fixture,
        label: `${fixture.label} / canvas ${mode}`,
        mode,
      }))
    ),
    {
      ...anima,
      label: `${anima.label} with inpaint adapter / canvas inpaint`,
      mode: 'inpaint',
      // The adapter switch is on by default, so only a panel that has it off shows a record turning it on.
      panelOverrides: { animaInpaintAdapterEnabled: false },
      settings: {
        ...anima.settings,
        animaInpaintAdapterEnabled: true,
        animaInpaintAdapterModel: ANIMA_INPAINT_ADAPTER,
        animaInpaintAdapterWeight: 0.7,
      },
    },
    {
      ...sdxl,
      compile: {
        canvasMetadata: { version: 1 },
        controlLayers: [
          {
            beginEndStepPct: [0.1, 0.85],
            controlMode: 'more_control',
            id: 'control',
            imageName: 'control.png',
            kind: 'controlnet',
            model: { base: 'sdxl', key: 'sdxl-controlnet', name: 'SDXL ControlNet', type: 'controlnet' },
            weight: 0.65,
          },
          {
            beginEndStepPct: [0, 1],
            controlMode: null,
            id: 't2i',
            imageName: 't2i.png',
            kind: 't2i_adapter',
            model: { base: 'sdxl', key: 'sdxl-t2i-adapter', name: 'SDXL T2I-Adapter', type: 't2i_adapter' },
            weight: 0.8,
          },
        ],
        regionalGuidance: [CANVAS_REGION],
      },
      label: `${sdxl.label} with layers / canvas img2img`,
      mode: 'img2img',
    },
    // A layout family places regions in its caption, which the record takes from the caption node.
    {
      ...flux3,
      compile: { regionalGuidance: [CANVAS_REGION] },
      label: `${flux3.label} with a region / canvas txt2img`,
      mode: 'txt2img',
    },
    // The provider was asked for its named size nearest the frame; a Canvas route keeps the frame.
    {
      ...gemini!,
      frame: { height: 800, width: 1000 },
      isCanvasFrame: true,
      label: `${gemini!.label} / canvas txt2img`,
      mode: 'txt2img',
    },
  ];
};

export const compileGenerateRecordCase = (
  { model, settings }: GenerateRecordCase,
  withSettings: GenerateSettings = settings
) => compileGenerateGraph(withSettings, model, 'gallery', PROJECT_SETTINGS);

/** The settings a Canvas case compiles: PiD decodes only the modes that keep the whole frame. */
export const canvasSettingsOf = ({ mode, settings }: CanvasRecordCase, withSettings: GenerateSettings = settings) =>
  mode === 'txt2img' || mode === 'img2img' ? withSettings : { ...withSettings, pidMode: 'off' as const };

export const compileCanvasRecordCase = (
  testCase: CanvasRecordCase,
  withSettings: GenerateSettings = testCase.settings
) => {
  const settings = canvasSettingsOf(testCase, withSettings);
  const { height, width } = testCase.frame ?? settings;

  return compileCanvas(testCase.model, testCase.mode, {
    ...testCase.compile,
    bbox: { height, width, x: 0, y: 0 },
    settings,
  });
};
