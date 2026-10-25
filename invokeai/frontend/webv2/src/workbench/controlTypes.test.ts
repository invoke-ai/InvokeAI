import type { ModelConfig } from '@features/models';

import { CONTROL_TYPES } from '@workbench/canvas-engine/api';
import { CONTROL_FILTERS } from '@workbench/canvas-operations/filterGraphs';
import { describe, expect, it } from 'vitest';

import {
  CONTROL_TYPE_SPECS,
  classifyControlModel,
  controlTypeForDetector,
  getDetectorFamily,
  PREPROCESSOR_FAMILY,
  type ControlModelClass,
} from './controlTypes';

describe('CONTROL_TYPE_SPECS', () => {
  it('covers every control type with detectors the filter catalog runs', () => {
    const filterTypes = new Set(CONTROL_FILTERS.map((filter) => filter.type));

    expect(Object.keys(CONTROL_TYPE_SPECS).sort()).toEqual([...CONTROL_TYPES].sort());
    for (const [type, spec] of Object.entries(CONTROL_TYPE_SPECS)) {
      expect(spec.id).toBe(type);
      expect(spec.detectors.filter((detector) => !filterTypes.has(detector.type))).toEqual([]);
      if (spec.defaultDetector !== null) {
        expect(spec.detectors.map((detector) => detector.type)).toContain(spec.defaultDetector);
      }
    }
  });

  it('maps every generic detector to the family its spec names, so model and detector families agree', () => {
    for (const spec of Object.values(CONTROL_TYPE_SPECS)) {
      for (const detector of spec.detectors.filter((entry) => !entry.settings)) {
        expect(PREPROCESSOR_FAMILY[detector.type], detector.type).toBe(detector.family);
      }
    }
  });
});

describe('getDetectorFamily', () => {
  it.each([
    ['depth', 'normal_map', 'normal'],
    ['depth', 'depth_anything_depth_estimation', 'depth'],
    ['depth', null, 'depth'],
    ['edges', 'mlsd_detection', 'mlsd'],
    ['sketch', 'pidi_edge_detection', 'scribble'],
    ['sketch', null, 'scribble'],
    ['tile', null, 'tile'],
    ['pose', 'canny_edge_detection', 'openpose'],
  ] as const)('%s with %s produces %s', (type, detector, family) => {
    expect(getDetectorFamily(type, detector)).toBe(family);
  });
});

describe('controlTypeForDetector', () => {
  it.each([
    [{ settings: { scribble: true }, type: 'pidi_edge_detection' }, 'sketch'],
    [{ settings: { scribble: false }, type: 'pidi_edge_detection' }, 'softedge'],
    [{ type: 'pidi_edge_detection' }, 'softedge'],
    [{ settings: {}, type: 'scribble_edge_detection' }, 'sketch'],
    [{ settings: {}, type: 'normal_map' }, 'depth'],
    [{ settings: {}, type: 'mediapipe_face_detection' }, 'pose'],
    [{ settings: {}, type: 'content_shuffle' }, 'color'],
    [{ settings: {}, type: 'img_blur' }, null],
  ])('%j belongs to %s', (detector, type) => {
    expect(controlTypeForDetector(detector)).toBe(type);
  });
});

const model = (
  name: string,
  base: string,
  type = 'controlnet',
  preprocessor: string | null = null
): Pick<ModelConfig, 'name' | 'base' | 'type' | 'default_settings'> =>
  ({ base, default_settings: preprocessor ? { preprocessor } : null, name, type }) as ModelConfig;

const made = (purpose: ControlModelClass['purpose'], family: ControlModelClass['family']): ControlModelClass => ({
  covers: [],
  family,
  purpose,
  union: false,
});

const FUN_COVERS = ['canny', 'mlsd', 'softedge', 'depth', 'openpose'];

describe('classifyControlModel', () => {
  it.each([
    [
      'legacy preprocessor id',
      model('My ControlNet', 'sdxl', 'controlnet', 'depth_anything_image_processor'),
      made('depth', 'depth'),
    ],
    [
      'legacy zoe id',
      model('My ControlNet', 'sdxl', 'controlnet', 'zoe_depth_image_processor'),
      made('depth', 'depth'),
    ],
    ['node id', model('My ControlNet', 'sdxl', 'controlnet', 'normal_map'), made('depth', 'normal')],
    ['legacy mlsd id', model('Lines', 'sd-1', 'controlnet', 'mlsd_image_processor'), made('edges', 'mlsd')],
    ['tile preprocessor', model('Custom', 'sdxl', 'controlnet', 'tile'), made('tile', 'tile')],
    ['tile name', model('xinsir-controlnet-tile-sdxl', 'sdxl'), made('tile', 'tile')],
    [
      'scribble name over the backend lineart guess',
      model('control_v11p_sd15_scribble', 'sd-1', 'controlnet', 'lineart_image_processor'),
      made('sketch', 'scribble'),
    ],
    [
      'anime line art over the preprocessor',
      model('control_v11p_sd15s2_lineart_anime', 'sd-1', 'controlnet', 'lineart_image_processor'),
      made('lineart', 'lineart_anime'),
    ],
    ['face name', model('controlnet mediapipe face', 'sd-1'), made('pose', 'face')],
    ['name word', model('control_v11p_sd15_canny', 'sd-1'), made('edges', 'canny')],
    ['name word prefix', model('control_v11p_sd15_normalbae', 'sd-1'), made('depth', 'normal')],
    ['camelCase name', model('controlnetSoftEdge', 'sdxl'), made('softedge', 'softedge')],
    ['openpose name', model('controlnet-openpose-sdxl-1.0', 'sdxl'), made('pose', 'openpose')],
    ['no match inside unrelated words', model('cached-surface-model', 'sdxl'), made(null, null)],
    ['unknown model', model('my-sdxl-control', 'sdxl'), made(null, null)],
    ['T2I-Adapter', model('t2iadapter_canny_sd15v2', 'sd-1', 't2i_adapter'), made('edges', 'canny')],
    ['Control LoRA', model('flux1-depth-dev-lora', 'flux', 'control_lora'), made('depth', 'depth')],
    ['Anima LLLite', model('anima-lllite-sketch', 'anima', 'controlnet'), made('sketch', 'scribble')],
    [
      'saved preprocessor over a weak name word',
      model('depth-something', 'sdxl', 'controlnet', 'canny_image_processor'),
      made('edges', 'canny'),
    ],
  ] as const)('%s', (_label, input, expected) => {
    expect(classifyControlModel(input)).toEqual(expected);
  });

  it.each([
    [
      'an SDXL union, covering the xinsir ProMax conditions',
      model('xinsir-controlnet-union-sdxl-promax', 'sdxl'),
      ['openpose', 'depth', 'canny', 'mlsd', 'lineart', 'lineart_anime', 'scribble', 'softedge', 'normal', 'tile'],
    ],
    ['an SD1.5 union, offered only as an explicit pick', model('controlnet-union-sd15', 'sd-1'), []],
    [
      'a FLUX union, covering the InstantX modes',
      model('FLUX.1-dev-Controlnet-Union-Pro', 'flux', 'controlnet', 'canny_image_processor'),
      ['canny', 'tile', 'depth', 'openpose'],
    ],
    ['a Z-Image Fun union whatever its name', model('Z-Image-Turbo-Fun-Controlnet', 'z-image'), FUN_COVERS],
    ['a Qwen-Image Fun union', model('Qwen-Image-2512-Fun-Controlnet-Union', 'qwen-image'), FUN_COVERS],
    ['a FLUX.2 Fun union', model('FLUX.2-dev-Fun-Controlnet', 'flux2'), FUN_COVERS],
  ] as const)('classifies %s', (_label, input, covers) => {
    expect(classifyControlModel(input)).toEqual({ covers, family: null, purpose: null, union: true });
  });

  it('classifies a replaced model config afresh', () => {
    const config = model('control_v11p_sd15_canny', 'sd-1');
    expect(classifyControlModel(config).family).toBe('canny');

    // A renamed model arrives as a new config object; the class must follow the new name.
    expect(classifyControlModel({ ...config, name: 'control_v11f1p_sd15_depth' }).family).toBe('depth');
    expect(classifyControlModel(config).family).toBe('canny');
  });
});
