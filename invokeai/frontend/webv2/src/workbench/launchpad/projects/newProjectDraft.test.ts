import type { GalleryImage } from '@features/gallery';
import type { ModelConfig } from '@features/models';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import {
  buildNewProjectRequest,
  commitDimension,
  createNewProjectDraft,
  getActiveAspectRatio,
  listSelectableGenerateModels,
  resolveNewProjectModel,
  resolveNewProjectSize,
  selectAspectRatio,
  selectModel,
  setCanvasStart,
  setStartImage,
  swapDimensions,
} from './newProjectDraft';

seedArchitectureCapabilities();

const model = (overrides: Partial<ModelConfig> & Pick<ModelConfig, 'key' | 'name'>): ModelConfig => ({
  base: 'sdxl',
  file_size: 1,
  format: 'checkpoint',
  hash: `hash-${overrides.key}`,
  path: `/models/${overrides.key}`,
  source: overrides.key,
  source_type: 'path',
  type: 'main',
  ...overrides,
});

const sdxlPortrait = model({
  default_settings: { height: 1216, width: 832 },
  key: 'sdxl-portrait',
  name: 'Zeta Portrait XL',
});
const sdxlPlain = model({ key: 'sdxl-plain', name: 'Alpha XL' });
const sd1 = model({ base: 'sd-1', key: 'sd1', name: 'Dreamshaper' });
const flux = model({ base: 'flux', format: 'diffusers', key: 'flux-dev', name: 'FLUX Dev' });
const lora = model({ format: 'lycoris', key: 'lora', name: 'Lora', type: 'lora' });
const unsupportedMain = model({ base: 'sdxl-refiner', key: 'refiner', name: 'Refiner' });

const catalog = [lora, sdxlPortrait, unsupportedMain, sd1, flux, sdxlPlain];
const selectable = listSelectableGenerateModels(catalog);

describe('listSelectableGenerateModels', () => {
  it('keeps only generate-capable mains, sorted by name', () => {
    expect(selectable.map((entry) => entry.key)).toEqual(['sdxl-plain', 'sd1', 'flux-dev', 'sdxl-portrait']);
  });
});

describe('resolveNewProjectModel', () => {
  it('prefers the session pick, then the remembered model, then the first selectable one', () => {
    const draft = createNewProjectDraft('generate');

    expect(resolveNewProjectModel(draft, selectable, null)?.key).toBe('sdxl-plain');
    expect(resolveNewProjectModel(draft, selectable, 'sd1')?.key).toBe('sd1');
    expect(resolveNewProjectModel(draft, selectable, 'uninstalled')?.key).toBe('sdxl-plain');
    expect(resolveNewProjectModel({ ...draft, modelKey: 'sdxl-plain' }, selectable, 'sd1')?.key).toBe('sdxl-plain');
    expect(resolveNewProjectModel(draft, [], 'sd1')).toBeNull();
  });
});

describe('size', () => {
  const draft = createNewProjectDraft('generate');
  const portrait = selectable.find((entry) => entry.key === 'sdxl-portrait')!;
  const sdxl = selectable.find((entry) => entry.key === 'sdxl-plain')!;
  const fluxDev = selectable.find((entry) => entry.key === 'flux-dev')!;
  const sd1Model = selectable.find((entry) => entry.key === 'sd1')!;

  it('follows the model defaults until the user sets a size', () => {
    expect(resolveNewProjectSize(draft, portrait)).toEqual({ height: 1216, width: 832 });
    expect(resolveNewProjectSize(draft, sd1Model)).toEqual({ height: 512, width: 512 });
    expect(resolveNewProjectSize(selectModel(draft, portrait), sdxl)).toEqual({ height: 1024, width: 1024 });
  });

  it('reshapes the model pixel budget to a chosen ratio', () => {
    // SDXL: 1024² at 3:2 → 1254×836, snapped to the 8px grid.
    expect(resolveNewProjectSize(selectAspectRatio(draft, '3:2', sdxl), sdxl)).toEqual({ height: 840, width: 1256 });
    expect(resolveNewProjectSize(selectAspectRatio(draft, '16:9', sdxl), sdxl)).toEqual({ height: 768, width: 1368 });
    // SD 1: 512² at 3:2 → 627×418, snapped to 8.
    expect(resolveNewProjectSize(selectAspectRatio(draft, '3:2', sd1Model), sd1Model)).toEqual({
      height: 416,
      width: 624,
    });
  });

  it('keeps a typed size across a model change, re-snapped to the new grid', () => {
    const typed = commitDimension(commitDimension(draft, sdxl, 'width', 1256), sdxl, 'height', 840);

    // 1256×840 on FLUX's 16px grid rounds up to 1264×848.
    expect(resolveNewProjectSize(selectModel(typed, fluxDev), fluxDev)).toEqual({ height: 848, width: 1264 });
  });

  it('carries a chip across a model change at the new model’s budget, chip still lit', () => {
    const sd1Chip = selectAspectRatio(draft, '3:2', sd1Model);

    expect(resolveNewProjectSize(sd1Chip, sd1Model)).toEqual({ height: 416, width: 624 });

    const moved = selectModel(sd1Chip, sdxl);

    expect(resolveNewProjectSize(moved, sdxl)).toEqual({ height: 840, width: 1256 });
    expect(getActiveAspectRatio(moved, resolveNewProjectSize(moved, sdxl))).toBe('3:2');
  });

  it('snaps typed dimensions to the grid within the bounds and ignores non-numbers', () => {
    expect(resolveNewProjectSize(commitDimension(draft, sdxl, 'width', 1003), sdxl)).toEqual({
      height: 1024,
      width: 1000,
    });
    expect(resolveNewProjectSize(commitDimension(draft, sdxl, 'height', 70_000), sdxl).height).toBe(4096);
    expect(resolveNewProjectSize(commitDimension(draft, sdxl, 'height', 10), sdxl).height).toBe(64);
    expect(commitDimension(draft, sdxl, 'width', Number.NaN)).toBe(draft);
  });

  it('swaps width and height and moves the active chip to the inverse ratio', () => {
    const landscape = selectAspectRatio(draft, '3:2', sdxl);
    const swapped = swapDimensions(landscape, sdxl);

    expect(resolveNewProjectSize(swapped, sdxl)).toEqual({ height: 1256, width: 840 });
    expect(getActiveAspectRatio(landscape, resolveNewProjectSize(landscape, sdxl))).toBe('3:2');
    expect(getActiveAspectRatio(swapped, resolveNewProjectSize(swapped, sdxl))).toBe('2:3');
  });

  it('lights the chip a default or typed size matches exactly, and none otherwise', () => {
    expect(getActiveAspectRatio(draft, resolveNewProjectSize(draft, sdxl))).toBe('1:1');
    expect(getActiveAspectRatio(draft, resolveNewProjectSize(draft, portrait))).toBeNull();

    const typed = commitDimension(selectAspectRatio(draft, '3:2', sdxl), sdxl, 'width', 1000);

    expect(getActiveAspectRatio(typed, resolveNewProjectSize(typed, sdxl))).toBeNull();
    expect(resolveNewProjectSize(typed, sdxl)).toEqual({ height: 840, width: 1000 });

    const square = commitDimension(typed, sdxl, 'height', 1000);

    expect(getActiveAspectRatio(square, resolveNewProjectSize(square, sdxl))).toBe('1:1');
  });
});

describe('buildNewProjectRequest', () => {
  const sdxl = selectable.find((entry) => entry.key === 'sdxl-plain')!;

  it('carries the model defaults with the chosen size for Generate and Canvas starts', () => {
    const draft = selectAspectRatio({ ...createNewProjectDraft('canvas'), name: '  Portraits ' }, '3:2', sdxl);
    const request = buildNewProjectRequest({ draft, model: sdxl, models: catalog });

    expect(request.intent).toBe('canvas');
    expect(request.name).toBe('Portraits');
    expect(request.generateSettings).toMatchObject({
      aspectRatioId: '3:2',
      aspectRatioIsLocked: false,
      height: 840,
      modelKey: 'sdxl-plain',
      scheduler: 'euler_a',
      steps: 30,
      width: 1256,
    });
    expect(request.generateSettings?.aspectRatioValue).toBe(1.5);
  });

  it('derives the ratio from a typed size', () => {
    const draft = commitDimension(createNewProjectDraft('generate'), sdxl, 'width', 1000);
    const { generateSettings } = buildNewProjectRequest({ draft, model: sdxl, models: catalog });

    expect(generateSettings).toMatchObject({ aspectRatioId: 'Free', height: 1024, width: 1000 });
    expect(generateSettings?.aspectRatioValue).toBeCloseTo(1000 / 1024);
  });

  it('keeps the default draft name for a blank name and no settings for other starts', () => {
    expect(buildNewProjectRequest({ draft: createNewProjectDraft('workflow'), model: sdxl, models: catalog })).toEqual({
      generateSettings: null,
      intent: 'workflow',
      libraryWorkflow: null,
      name: null,
      startImage: null,
    });
    expect(
      buildNewProjectRequest({ draft: createNewProjectDraft('generate'), model: null, models: [] }).generateSettings
    ).toBeNull();
  });

  it('sizes a Canvas start to its image, snapped to the grid, and carries the image', () => {
    const image: GalleryImage = {
      boardId: 'none',
      height: 750,
      imageCategory: 'user',
      imageName: 'pasted.png',
      imageUrl: '/images/pasted.png',
      queuedAt: '',
      sourceQueueItemId: '',
      starred: false,
      thumbnailUrl: '/thumbs/pasted.png',
      width: 1000,
    };
    const withImage = setStartImage(selectAspectRatio(createNewProjectDraft('canvas'), '3:2', sdxl), image);

    expect(resolveNewProjectSize(withImage, sdxl)).toEqual({ height: 752, width: 1000 });
    expect(getActiveAspectRatio(withImage, resolveNewProjectSize(withImage, sdxl))).toBeNull();
    const request = buildNewProjectRequest({ draft: withImage, model: sdxl, models: catalog });

    // The ratio follows the image itself (1000×750 is exactly 4:3), not the snapped render size.
    expect(request).toMatchObject({
      generateSettings: { aspectRatioId: '4:3', aspectRatioValue: 4 / 3, height: 752, width: 1000 },
      startImage: image,
    });

    // Blank forgets the image and returns to the size chosen before it.
    const blank = setCanvasStart(withImage, 'blank');

    expect(blank.startImage).toBeNull();
    expect(resolveNewProjectSize(blank, sdxl)).toEqual({ height: 840, width: 1256 });
    expect(buildNewProjectRequest({ draft: blank, model: sdxl, models: catalog }).startImage).toBeNull();

    // The image belongs to Canvas starts: a Generate start ignores it.
    const generate = { ...withImage, intent: 'generate' as const };

    expect(resolveNewProjectSize(generate, sdxl)).toEqual({ height: 840, width: 1256 });
    expect(buildNewProjectRequest({ draft: generate, model: sdxl, models: catalog }).startImage).toBeNull();
  });

  it('carries a picked library workflow for Workflow starts only', () => {
    const libraryWorkflow = { name: 'Upscale pipeline', workflowId: 'lib-1' };

    expect(
      buildNewProjectRequest({
        draft: { ...createNewProjectDraft('workflow'), libraryWorkflow },
        model: null,
        models: [],
      })
    ).toMatchObject({ generateSettings: null, intent: 'workflow', libraryWorkflow });
    expect(
      buildNewProjectRequest({
        draft: { ...createNewProjectDraft('generate'), libraryWorkflow },
        model: sdxl,
        models: catalog,
      }).libraryWorkflow
    ).toBeNull();
  });
});
