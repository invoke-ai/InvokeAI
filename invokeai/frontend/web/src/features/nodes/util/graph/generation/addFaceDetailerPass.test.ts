import type { RootState } from 'app/store/store';
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
    });
    expect(segmentAnything).toMatchObject({
      model: 'segment-anything-2-small',
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
    expect(g.getMetadataNode()).toMatchObject({
      detailer_enabled: true,
      detailer_version: 'v3',
      detailer_type: 'target',
      detailer_detector: 'grounding-dino-sam',
      detailer_quality: 'balanced',
      detailer_target_prompt: 'hands',
      detailer_upscale_method: 'pixel_crop_resize',
      detailer_face_selection: 'largest_area',
      detailer_face_id: 2,
      detailer_detection_threshold: 0.2,
      detailer_target_size: 640,
      detailer_max_upscale: 3,
      detailer_max_process_size: 896,
      detailer_crop_padding: 72,
      detailer_denoise_mask_expand: 10,
      detailer_denoise_mask_feather: 8,
      detailer_paste_mask_expand: 2,
      detailer_paste_mask_feather: 12,
      detailer_color_correct_mode: 'off',
      detailer_strength: 0.4,
      detailer_steps: 14,
      detailer_cfg_scale: 6.5,
    });
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

  it('adds detailer color correction when explicitly enabled', () => {
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
    expect(hasEdge(graph, crop.id, 'denoise_mask', colorCorrect.id, 'mask')).toBe(true);
    expect(hasEdge(graph, colorCorrect.id, 'image', paste.id, 'image')).toBe(true);
  });

  it('keeps the legacy MediaPipe detailer path available', () => {
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
