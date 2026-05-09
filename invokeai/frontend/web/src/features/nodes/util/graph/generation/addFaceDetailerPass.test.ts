import type { RootState } from 'app/store/store';
import {
  getDetailerTargetProfile,
  getGroundedSamDetailerRuntimeConfig,
  getGroundedSamDetectorPromptConfig,
} from 'features/controlLayers/store/detailerRuntimeConfig';
import { getInitialParamsState } from 'features/controlLayers/store/types';
import type { GraphType } from 'features/nodes/util/graph/generation/Graph';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';
import { describe, expect, it } from 'vitest';

import { addFaceDetailerPass } from './addFaceDetailerPass';

const model = {
  key: 'model',
  hash: 'hash',
  name: 'Model',
  base: 'sdxl',
  type: 'main',
};

const makeState = (params?: Partial<RootState['params']>): RootState =>
  ({
    gallery: {
      autoAddBoardId: 'none',
    },
    params: {
      ...getInitialParamsState(),
      model,
      ...params,
    },
  }) as unknown as RootState;

type TestNode<T extends string> = {
  id: string;
  type: T;
  [key: string]: unknown;
};

const getNodesByType = <T extends string>(graph: GraphType, type: T): TestNode<T>[] =>
  Object.values(graph.nodes)
    .filter((node) => node.type === type)
    .map((node) => node as unknown as TestNode<T>);

const hasEdge = (
  graph: GraphType,
  sourceId: string,
  sourceField: string,
  destinationId: string,
  destinationField: string
) =>
  graph.edges.some(
    (edge) =>
      edge.source.node_id === sourceId &&
      edge.source.field === sourceField &&
      edge.destination.node_id === destinationId &&
      edge.destination.field === destinationField
  );

const buildBaseGraph = () => {
  const g = new Graph('face-detailer-test');
  const modelLoader = g.addNode({
    id: 'model_loader',
    type: 'sdxl_model_loader',
    model: model as Invocation<'sdxl_model_loader'>['model'],
  });
  const baseDenoise = g.addNode({
    id: 'base_denoise',
    type: 'denoise_latents',
  });
  const baseImage = g.addNode({
    id: 'base_l2i',
    type: 'l2i',
    fp32: true,
  });
  const posCondCollect = g.addNode({
    id: 'pos_cond_collect',
    type: 'collect',
  });
  const negCondCollect = g.addNode({
    id: 'neg_cond_collect',
    type: 'collect',
  });
  const seed = g.addNode({
    id: 'seed',
    type: 'integer',
  });

  g.addEdge(modelLoader, 'unet', baseDenoise, 'unet');

  return { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed };
};

describe('detailer runtime config', () => {
  it('uses stored params exactly for localized targets', () => {
    const params = {
      ...getInitialParamsState(),
      detailerQuality: 'balanced',
      detailerTargetSize: 640,
      detailerMaxUpscale: 3,
      detailerMaxProcessSize: 896,
      detailerDenoiseMaskExpand: 10,
      detailerDenoiseMaskFeather: 8,
      detailerPasteMaskExpand: 2,
      detailerPasteMaskFeather: 12,
      detailerStrength: 0.4,
      detailerSteps: 22,
      detailerCfgScale: 6.5,
    } as const;

    expect(getDetailerTargetProfile('hands')).toBe('localized');
    expect(getGroundedSamDetailerRuntimeConfig(params, 'hands')).toMatchObject({
      targetProfile: 'localized',
      targetSize: 640,
      maxUpscale: 3,
      maxProcessSize: 896,
      denoiseMaskExpand: 10,
      denoiseMaskFeather: 8,
      pasteMaskExpand: 2,
      pasteMaskFeather: 12,
      denoiseMaskContract: 0,
      pasteMaskContract: 0,
      preventDownscale: false,
      cfgScale: 6.5,
      steps: 22,
      strength: 0.4,
      edgeRadius: 8,
    });
  });

  it('uses the Body/person preservation profile without mutating stored params', () => {
    const params = {
      ...getInitialParamsState(),
      detailerQuality: 'fast',
      detailerTargetSize: 512,
      detailerMaxUpscale: 4,
      detailerMaxProcessSize: 512,
      detailerDenoiseMaskExpand: 8,
      detailerDenoiseMaskFeather: 4,
      detailerPasteMaskExpand: 0,
      detailerPasteMaskFeather: 6,
      detailerStrength: 0.24,
      detailerSteps: 8,
      detailerCfgScale: 9,
    } as const;

    expect(getDetailerTargetProfile('Full Body')).toBe('person');
    expect(getGroundedSamDetailerRuntimeConfig(params, 'Full Body')).toMatchObject({
      targetProfile: 'person',
      targetSize: 1024,
      maxUpscale: 4,
      maxProcessSize: 1024,
      denoiseMaskExpand: 0,
      denoiseMaskFeather: 2,
      pasteMaskExpand: 0,
      pasteMaskFeather: 4,
      denoiseMaskContract: 4,
      pasteMaskContract: 2,
      preventDownscale: true,
      cfgScale: 4.5,
      steps: 8,
      strength: 0.1,
      edgeRadius: 2,
    });
    expect(params.detailerTargetSize).toBe(512);
    expect(params.detailerMaxProcessSize).toBe(512);
    expect(params.detailerCfgScale).toBe(9);
  });

  it('maps user-facing target presets to detector prompts and label priorities', () => {
    expect(getGroundedSamDetectorPromptConfig('face')).toEqual({
      detectorPrompt: 'face',
      labelPriority: 'face',
    });
    expect(getGroundedSamDetectorPromptConfig('head')).toEqual({
      detectorPrompt: 'head',
      labelPriority: 'head',
    });
    expect(getGroundedSamDetectorPromptConfig('hands')).toEqual({
      detectorPrompt: 'hands',
      labelPriority: 'hands',
    });
    expect(getGroundedSamDetectorPromptConfig('person')).toEqual({
      detectorPrompt: 'person',
      labelPriority: 'person',
    });
    expect(getGroundedSamDetectorPromptConfig('belt | buckle')).toEqual({
      detectorPrompt: 'belt | buckle',
      labelPriority: 'belt | buckle',
    });
  });
});

describe('addFaceDetailerPass', () => {
  it('returns the original image and does not add detailer nodes when disabled', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    const result = addFaceDetailerPass({
      g,
      state: makeState({ detailerEnabled: false }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
    });

    const graph = g.getGraph();

    expect(result).toBe(baseImage);
    expect(getNodesByType(graph, 'face_off')).toHaveLength(0);
    expect(getNodesByType(graph, 'grounding_dino')).toHaveLength(0);
    expect(getNodesByType(graph, 'img_paste')).toHaveLength(0);
    expect(getNodesByType(graph, 'detailer_paste_crop')).toHaveLength(0);
    expect(getNodesByType(graph, 'color_correct')).toHaveLength(0);
    expect(getNodesByType(graph, 'detailer_debug_collage')).toHaveLength(0);
  });

  it('returns the original image and does not add detailer nodes for unsupported model bases', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    const result = addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        model: { ...model, base: 'flux' } as RootState['params']['model'],
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
    });

    const graph = g.getGraph();

    expect(result).toBe(baseImage);
    expect(getNodesByType(graph, 'face_off')).toHaveLength(0);
    expect(getNodesByType(graph, 'grounding_dino')).toHaveLength(0);
    expect(getNodesByType(graph, 'img_paste')).toHaveLength(0);
    expect(getNodesByType(graph, 'detailer_paste_crop')).toHaveLength(0);
    expect(getNodesByType(graph, 'color_correct')).toHaveLength(0);
    expect(getNodesByType(graph, 'detailer_debug_collage')).toHaveLength(0);
  });

  it('adds DINO, SAM, crop prep, second denoise, and mask-safe paste-back nodes when enabled', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    const result = addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        detailerTargetPrompt: 'hands',
        detailerFaceSelection: 'largest_area',
        detailerFaceId: 2,
        detailerDetectionThreshold: 0.2,
        detailerTargetSize: 640,
        detailerMaxUpscale: 3,
        detailerMaxProcessSize: 896,
        detailerCropPadding: 72,
        detailerDenoiseMaskExpand: 10,
        detailerDenoiseMaskFeather: 8,
        detailerPasteMaskExpand: 2,
        detailerPasteMaskFeather: 12,
        detailerStrength: 0.4,
        detailerSteps: 14,
        detailerCfgScale: 6.5,
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
      colorCompensation: 'SDXL',
    });

    const graph = g.getGraph();
    const groundingDino = getNodesByType(graph, 'grounding_dino')[0];
    const selectBoundingBox = getNodesByType(graph, 'select_bounding_box')[0];
    const segmentAnything = getNodesByType(graph, 'segment_anything')[0];
    const tensorMaskToImage = getNodesByType(graph, 'tensor_mask_to_image')[0];
    const crop = getNodesByType(graph, 'detailer_crop_from_mask')[0];
    const paste = getNodesByType(graph, 'detailer_paste_crop')[0];
    const detailNoise = getNodesByType(graph, 'noise')[0];
    const detailI2L = getNodesByType(graph, 'i2l')[0];
    const detailL2I = getNodesByType(graph, 'l2i').find((node) => node.id !== baseImage.id);
    const detailDenoise = getNodesByType(graph, 'denoise_latents').find((node) => node.id !== baseDenoise.id);
    const createGradientMask = getNodesByType(graph, 'create_gradient_mask')[0];

    assert(groundingDino);
    assert(selectBoundingBox);
    assert(segmentAnything);
    assert(tensorMaskToImage);
    assert(crop);
    assert(paste);
    assert(detailNoise);
    assert(detailI2L);
    assert(detailL2I);
    assert(detailDenoise);
    assert(createGradientMask);

    expect(result).toBe(paste);
    expect(groundingDino).toMatchObject({
      model: 'grounding-dino-tiny',
      prompt: 'hands',
      detection_threshold: 0.2,
    });
    expect(selectBoundingBox).toMatchObject({
      selection_mode: 'largest_area',
      index: 2,
      label_priority: 'hands',
    });
    expect(segmentAnything).toMatchObject({
      model: 'segment-anything-2-base',
      apply_polygon_refinement: true,
      mask_filter: 'all',
    });
    expect(crop).toMatchObject({
      padding: 72,
      denoise_mask_expand: 10,
      denoise_mask_feather: 8,
      paste_mask_expand: 2,
      paste_mask_feather: 12,
      target_size: 640,
      max_upscale: 3,
      max_process_size: 896,
      prevent_downscale: false,
      denoise_mask_contract: 0,
      paste_mask_contract: 0,
    });
    expect(detailI2L).toMatchObject({
      color_compensation: 'SDXL',
    });
    expect(detailDenoise).toMatchObject({
      cfg_scale: 6.5,
      steps: 14,
      denoising_end: 1,
    });
    expect(detailDenoise?.denoising_start).toBeCloseTo(0.6);
    expect(createGradientMask).toMatchObject({
      edge_radius: 8,
      minimum_denoise: 0,
    });
    expect(hasEdge(graph, baseImage.id, 'image', groundingDino.id, 'image')).toBe(true);
    expect(hasEdge(graph, groundingDino.id, 'collection', selectBoundingBox.id, 'collection')).toBe(true);
    expect(hasEdge(graph, selectBoundingBox.id, 'collection', segmentAnything.id, 'bounding_boxes')).toBe(true);
    expect(hasEdge(graph, segmentAnything.id, 'mask', tensorMaskToImage.id, 'mask')).toBe(true);
    expect(hasEdge(graph, tensorMaskToImage.id, 'image', crop.id, 'mask')).toBe(true);
    expect(hasEdge(graph, crop.id, 'image', detailI2L.id, 'image')).toBe(true);
    expect(hasEdge(graph, crop.id, 'processed_width', detailNoise.id, 'width')).toBe(true);
    expect(hasEdge(graph, crop.id, 'processed_height', detailNoise.id, 'height')).toBe(true);
    expect(hasEdge(graph, crop.id, 'denoise_mask', createGradientMask.id, 'mask')).toBe(true);
    expect(hasEdge(graph, posCondCollect.id, 'collection', detailDenoise.id, 'positive_conditioning')).toBe(true);
    expect(hasEdge(graph, negCondCollect.id, 'collection', detailDenoise.id, 'negative_conditioning')).toBe(true);
    expect(hasEdge(graph, detailDenoise.id, 'latents', detailL2I.id, 'latents')).toBe(true);
    expect(hasEdge(graph, detailL2I.id, 'image', paste.id, 'image')).toBe(true);
    expect(hasEdge(graph, crop.id, 'paste_alpha_mask', paste.id, 'paste_alpha_mask')).toBe(true);
    expect(hasEdge(graph, crop.id, 'x', paste.id, 'x')).toBe(true);
    expect(hasEdge(graph, crop.id, 'y', paste.id, 'y')).toBe(true);
    expect(hasEdge(graph, crop.id, 'original_width', paste.id, 'original_width')).toBe(true);
    expect(hasEdge(graph, crop.id, 'original_height', paste.id, 'original_height')).toBe(true);
    expect(getNodesByType(graph, 'img_paste')).toHaveLength(0);
    expect(getNodesByType(graph, 'img_resize')).toHaveLength(0);
    expect(getNodesByType(graph, 'color_correct')).toHaveLength(0);
    expect(getNodesByType(graph, 'detailer_debug_collage')).toHaveLength(0);
    expect(g.getMetadataNode()).toMatchObject({
      detailer_enabled: true,
      detailer_version: 'v5',
      detailer_type: 'target',
      detailer_detector: 'grounding-dino-sam',
      detailer_quality: 'balanced',
      detailer_target_profile: 'localized',
      detailer_target_prompt: 'hands',
      detailer_detector_prompt: 'hands',
      detailer_upscale_method: 'pixel_crop_resize',
      detailer_face_selection: 'largest_area',
      detailer_face_id: 2,
      detailer_detection_threshold: 0.2,
      detailer_sam_model: 'segment-anything-2-base',
      detailer_target_size: 640,
      detailer_param_target_size: 640,
      detailer_max_upscale: 3,
      detailer_param_max_upscale: 3,
      detailer_max_process_size: 896,
      detailer_param_max_process_size: 896,
      detailer_crop_padding: 72,
      detailer_denoise_mask_expand: 10,
      detailer_param_denoise_mask_expand: 10,
      detailer_denoise_mask_feather: 8,
      detailer_param_denoise_mask_feather: 8,
      detailer_paste_mask_expand: 2,
      detailer_param_paste_mask_expand: 2,
      detailer_paste_mask_feather: 12,
      detailer_param_paste_mask_feather: 12,
      detailer_denoise_mask_contract: 0,
      detailer_paste_mask_contract: 0,
      detailer_prevent_downscale: false,
      detailer_color_correct_mode: 'off',
      detailer_strength: 0.4,
      detailer_param_strength: 0.4,
      detailer_steps: 14,
      detailer_param_steps: 14,
      detailer_cfg_scale: 6.5,
      detailer_param_cfg_scale: 6.5,
    });
  });

  it('uses a single-term detector prompt for the Head target while preserving the stored target', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        detailerTargetPrompt: 'head',
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
    });

    const graph = g.getGraph();
    const groundingDino = getNodesByType(graph, 'grounding_dino')[0];
    const selectBoundingBox = getNodesByType(graph, 'select_bounding_box')[0];
    const segmentAnything = getNodesByType(graph, 'segment_anything')[0];

    assert(groundingDino);
    assert(selectBoundingBox);
    assert(segmentAnything);

    expect(groundingDino).toMatchObject({
      prompt: 'head',
    });
    expect(selectBoundingBox).toMatchObject({
      label_priority: 'head',
    });
    expect(segmentAnything).toMatchObject({
      model: 'segment-anything-2-base',
    });
    expect(g.getMetadataNode()).toMatchObject({
      detailer_target_prompt: 'head',
      detailer_detector_prompt: 'head',
      detailer_sam_model: 'segment-anything-2-base',
    });
  });

  it('applies the conservative person runtime profile without mutating persisted params', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();
    const state = makeState({
      detailerEnabled: true,
      detailerQuality: 'balanced',
      detailerTargetPrompt: 'person',
      detailerTargetSize: 640,
      detailerMaxUpscale: 3,
      detailerMaxProcessSize: 896,
      detailerDenoiseMaskExpand: 10,
      detailerDenoiseMaskFeather: 8,
      detailerPasteMaskExpand: 2,
      detailerPasteMaskFeather: 12,
      detailerStrength: 0.4,
      detailerSteps: 22,
      detailerCfgScale: 9,
    });

    addFaceDetailerPass({
      g,
      state,
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
    });

    const graph = g.getGraph();
    const crop = getNodesByType(graph, 'detailer_crop_from_mask')[0];
    const groundingDino = getNodesByType(graph, 'grounding_dino')[0];
    const selectBoundingBox = getNodesByType(graph, 'select_bounding_box')[0];
    const segmentAnything = getNodesByType(graph, 'segment_anything')[0];
    const detailDenoise = getNodesByType(graph, 'denoise_latents').find((node) => node.id !== baseDenoise.id);
    const createGradientMask = getNodesByType(graph, 'create_gradient_mask')[0];

    assert(crop);
    assert(groundingDino);
    assert(selectBoundingBox);
    assert(segmentAnything);
    assert(detailDenoise);
    assert(createGradientMask);

    expect(groundingDino).toMatchObject({
      prompt: 'person',
    });
    expect(selectBoundingBox).toMatchObject({
      label_priority: 'person',
    });
    expect(segmentAnything).toMatchObject({
      model: 'segment-anything-2-base',
    });
    expect(crop).toMatchObject({
      denoise_mask_expand: 0,
      denoise_mask_feather: 2,
      paste_mask_expand: 0,
      paste_mask_feather: 4,
      target_size: 1024,
      max_upscale: 8,
      max_process_size: 1024,
      prevent_downscale: true,
      denoise_mask_contract: 4,
      paste_mask_contract: 2,
    });
    expect(detailDenoise).toMatchObject({
      cfg_scale: 4.5,
      steps: 14,
      denoising_end: 1,
    });
    expect(detailDenoise?.denoising_start).toBeCloseTo(0.86);
    expect(createGradientMask).toMatchObject({
      edge_radius: 2,
    });
    expect(g.getMetadataNode()).toMatchObject({
      detailer_target_profile: 'person',
      detailer_detector_prompt: 'person',
      detailer_sam_model: 'segment-anything-2-base',
      detailer_target_size: 1024,
      detailer_param_target_size: 640,
      detailer_max_upscale: 8,
      detailer_param_max_upscale: 3,
      detailer_max_process_size: 1024,
      detailer_param_max_process_size: 896,
      detailer_denoise_mask_expand: 0,
      detailer_param_denoise_mask_expand: 10,
      detailer_denoise_mask_feather: 2,
      detailer_param_denoise_mask_feather: 8,
      detailer_paste_mask_expand: 0,
      detailer_param_paste_mask_expand: 2,
      detailer_paste_mask_feather: 4,
      detailer_param_paste_mask_feather: 12,
      detailer_denoise_mask_contract: 4,
      detailer_paste_mask_contract: 2,
      detailer_prevent_downscale: true,
      detailer_strength: 0.14,
      detailer_param_strength: 0.4,
      detailer_steps: 14,
      detailer_param_steps: 22,
      detailer_cfg_scale: 4.5,
      detailer_param_cfg_scale: 9,
    });
    expect(state.params.detailerTargetSize).toBe(640);
    expect(state.params.detailerMaxUpscale).toBe(3);
    expect(state.params.detailerMaxProcessSize).toBe(896);
    expect(state.params.detailerStrength).toBe(0.4);
    expect(state.params.detailerSteps).toBe(22);
    expect(state.params.detailerCfgScale).toBe(9);
  });

  it('uses the high person profile as a 1536px preservation pass', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        detailerQuality: 'high',
        detailerTargetPrompt: 'full body',
        detailerCfgScale: 9,
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
    });

    const graph = g.getGraph();
    const groundingDino = getNodesByType(graph, 'grounding_dino')[0];
    const segmentAnything = getNodesByType(graph, 'segment_anything')[0];
    const crop = getNodesByType(graph, 'detailer_crop_from_mask')[0];
    const detailDenoise = getNodesByType(graph, 'denoise_latents').find((node) => node.id !== baseDenoise.id);

    assert(groundingDino);
    assert(segmentAnything);
    assert(crop);
    assert(detailDenoise);

    expect(groundingDino).toMatchObject({
      prompt: 'full body',
    });
    expect(segmentAnything).toMatchObject({
      model: 'segment-anything-2-base',
    });
    expect(crop).toMatchObject({
      target_size: 1536,
      max_upscale: 12,
      max_process_size: 1536,
      prevent_downscale: true,
    });
    expect(detailDenoise).toMatchObject({
      cfg_scale: 5,
      steps: 18,
    });
    expect(detailDenoise?.denoising_start).toBeCloseTo(0.82);
    expect(g.getMetadataNode()).toMatchObject({
      detailer_sam_model: 'segment-anything-2-base',
    });
  });

  it('does not add debug collage output outside development even when persisted state is enabled', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    const result = addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        detailerDebugEnabled: true,
        detailerTargetPrompt: 'belt',
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
    });

    const graph = g.getGraph();
    const paste = getNodesByType(graph, 'detailer_paste_crop')[0];

    expect(result).toBe(paste);
    expect(getNodesByType(graph, 'detailer_debug_collage')).toHaveLength(0);
    expect(getNodesByType(graph, 'canvas_output')).toHaveLength(0);
  });

  it('adds a secondary debug collage canvas output when debug is enabled in development', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    const result = addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        detailerDebugEnabled: true,
        detailerTargetPrompt: 'belt',
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
      isDetailerDebugOutputAllowed: true,
    });

    const graph = g.getGraph();
    const selectBoundingBox = getNodesByType(graph, 'select_bounding_box')[0];
    const tensorMaskToImage = getNodesByType(graph, 'tensor_mask_to_image')[0];
    const crop = getNodesByType(graph, 'detailer_crop_from_mask')[0];
    const detailL2I = getNodesByType(graph, 'l2i').find((node) => node.id !== baseImage.id);
    const paste = getNodesByType(graph, 'detailer_paste_crop')[0];
    const debugCollage = getNodesByType(graph, 'detailer_debug_collage')[0];
    const debugCanvasOutput = getNodesByType(graph, 'canvas_output')[0];

    assert(selectBoundingBox);
    assert(tensorMaskToImage);
    assert(crop);
    assert(detailL2I);
    assert(paste);
    assert(debugCollage);
    assert(debugCanvasOutput);

    expect(result).toBe(paste);
    expect(debugCollage).toMatchObject({
      target_prompt: 'belt',
      detector_prompt: 'belt',
      sam_model: 'segment-anything-2-base',
    });
    expect(debugCanvasOutput).toMatchObject({
      is_intermediate: false,
      use_cache: false,
    });
    expect(hasEdge(graph, baseImage.id, 'image', debugCollage.id, 'base_image')).toBe(true);
    expect(hasEdge(graph, selectBoundingBox.id, 'collection', debugCollage.id, 'selected_bounding_boxes')).toBe(true);
    expect(hasEdge(graph, tensorMaskToImage.id, 'image', debugCollage.id, 'mask')).toBe(true);
    expect(hasEdge(graph, crop.id, 'image', debugCollage.id, 'processed_crop')).toBe(true);
    expect(hasEdge(graph, crop.id, 'denoise_mask', debugCollage.id, 'denoise_mask')).toBe(true);
    expect(hasEdge(graph, crop.id, 'paste_alpha_mask', debugCollage.id, 'paste_alpha_mask')).toBe(true);
    expect(hasEdge(graph, detailL2I.id, 'image', debugCollage.id, 'detailed_crop')).toBe(true);
    expect(hasEdge(graph, paste.id, 'image', debugCollage.id, 'final_image')).toBe(true);
    expect(hasEdge(graph, crop.id, 'x', debugCollage.id, 'x')).toBe(true);
    expect(hasEdge(graph, crop.id, 'y', debugCollage.id, 'y')).toBe(true);
    expect(hasEdge(graph, crop.id, 'original_width', debugCollage.id, 'original_width')).toBe(true);
    expect(hasEdge(graph, crop.id, 'original_height', debugCollage.id, 'original_height')).toBe(true);
    expect(hasEdge(graph, crop.id, 'processed_width', debugCollage.id, 'processed_width')).toBe(true);
    expect(hasEdge(graph, crop.id, 'processed_height', debugCollage.id, 'processed_height')).toBe(true);
    expect(hasEdge(graph, crop.id, 'detected', debugCollage.id, 'detected')).toBe(true);
    expect(hasEdge(graph, debugCollage.id, 'image', debugCanvasOutput.id, 'image')).toBe(true);
    expect(hasEdge(graph, g.getMetadataNode().id, 'metadata', debugCanvasOutput.id, 'metadata')).toBe(false);
  });

  it('skips detailer color correction when disabled', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        detailerColorCorrectMode: 'off',
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
    });

    const graph = g.getGraph();
    const paste = getNodesByType(graph, 'detailer_paste_crop')[0];
    const detailL2I = getNodesByType(graph, 'l2i').find((node) => node.id !== baseImage.id);

    assert(paste);
    assert(detailL2I);

    expect(getNodesByType(graph, 'color_correct')).toHaveLength(0);
    expect(hasEdge(graph, detailL2I.id, 'image', paste.id, 'image')).toBe(true);
  });

  it('wires detailer color correction through the denoise mask polarity contract', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        detailerColorCorrectMode: 'YCbCr-Luma',
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
    });

    const graph = g.getGraph();
    const crop = getNodesByType(graph, 'detailer_crop_from_mask')[0];
    const paste = getNodesByType(graph, 'detailer_paste_crop')[0];
    const detailL2I = getNodesByType(graph, 'l2i').find((node) => node.id !== baseImage.id);
    const colorCorrect = getNodesByType(graph, 'color_correct')[0];

    assert(crop);
    assert(paste);
    assert(detailL2I);
    assert(colorCorrect);

    expect(colorCorrect).toMatchObject({
      colorspace: 'YCbCr-Luma',
    });
    expect(hasEdge(graph, detailL2I.id, 'image', colorCorrect.id, 'base_image')).toBe(true);
    expect(hasEdge(graph, crop.id, 'image', colorCorrect.id, 'color_reference')).toBe(true);
    // detailer_crop_from_mask.denoise_mask is black = edit, white = preserve.
    // color_correct.mask is white = original, black = result, so this corrects only the edited target.
    expect(hasEdge(graph, crop.id, 'denoise_mask', colorCorrect.id, 'mask')).toBe(true);
    expect(hasEdge(graph, colorCorrect.id, 'image', paste.id, 'image')).toBe(true);
  });

  it('forces a persisted MediaPipe detector to DINO/SAM outside development', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    const result = addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        detailerDetector: 'mediapipe',
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
    });

    const graph = g.getGraph();
    const groundingDino = getNodesByType(graph, 'grounding_dino')[0];
    const crop = getNodesByType(graph, 'detailer_crop_from_mask')[0];
    const paste = getNodesByType(graph, 'detailer_paste_crop')[0];

    assert(groundingDino);
    assert(crop);
    assert(paste);

    expect(result).toBe(paste);
    expect(getNodesByType(graph, 'face_off')).toHaveLength(0);
    expect(getNodesByType(graph, 'img_paste')).toHaveLength(0);
    expect(g.getMetadataNode()).toMatchObject({
      detailer_enabled: true,
      detailer_detector: 'grounding-dino-sam',
    });
  });

  it('keeps the legacy MediaPipe detailer path available in development', () => {
    const { g, modelLoader, baseDenoise, baseImage, posCondCollect, negCondCollect, seed } = buildBaseGraph();

    const result = addFaceDetailerPass({
      g,
      state: makeState({
        detailerEnabled: true,
        detailerDetector: 'mediapipe',
        detailerFaceId: 2,
        detailerMinConfidence: 0.65,
        detailerPadding: 48,
        detailerStrength: 0.4,
        detailerSteps: 14,
        detailerCfgScale: 6.5,
        detailerMaskBlur: 12,
      }),
      image: baseImage,
      vaeSource: modelLoader,
      baseDenoise,
      posCondCollect,
      negCondCollect,
      seed,
      fp32: true,
      isLegacyDetailerDetectorAllowed: true,
    });

    const graph = g.getGraph();
    const faceOff = getNodesByType(graph, 'face_off')[0];
    const paste = getNodesByType(graph, 'img_paste')[0];
    const detailNoise = getNodesByType(graph, 'noise')[0];
    const detailL2I = getNodesByType(graph, 'l2i').find((node) => node.id !== baseImage.id);
    const detailDenoise = getNodesByType(graph, 'denoise_latents').find((node) => node.id !== baseDenoise.id);
    const createGradientMask = getNodesByType(graph, 'create_gradient_mask')[0];

    assert(faceOff);
    assert(paste);
    assert(detailNoise);
    assert(detailL2I);
    assert(detailDenoise);
    assert(createGradientMask);

    expect(result).toBe(paste);
    expect(faceOff).toMatchObject({
      face_id: 2,
      minimum_confidence: 0.65,
      padding: 48,
      chunk: false,
    });
    expect(createGradientMask).toMatchObject({
      edge_radius: 12,
      minimum_denoise: 0,
    });
    expect(hasEdge(graph, baseImage.id, 'image', faceOff.id, 'image')).toBe(true);
    expect(hasEdge(graph, faceOff.id, 'width', detailNoise.id, 'width')).toBe(true);
    expect(hasEdge(graph, detailL2I.id, 'image', paste.id, 'image')).toBe(true);
    expect(g.getMetadataNode()).toMatchObject({
      detailer_enabled: true,
      detailer_version: 'v1',
      detailer_detector: 'mediapipe',
    });
  });
});
