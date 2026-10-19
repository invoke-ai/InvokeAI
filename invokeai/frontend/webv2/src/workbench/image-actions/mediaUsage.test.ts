import type { GalleryItemRef } from '@features/gallery/contracts';
import type { CanvasNodeContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import { collectMediaUsage } from './mediaUsage';

const widget = (typeId: string, values: Record<string, unknown>) => ({
  createdAt: '2026-01-01T00:00:00.000Z',
  id: typeId,
  state: { id: typeId, label: typeId, values, version: 1 as const },
  typeId,
});

const rasterLayer = (id: string, imageName: string) =>
  ({
    id,
    isEnabled: true,
    isLocked: false,
    name: id,
    source: { image: { height: 64, imageName, width: 64 }, type: 'image' },
    type: 'raster',
  }) as unknown as CanvasNodeContract;

const project = ({
  stacks = {},
  widgets = {},
  workflowNodes = [],
}: {
  stacks?: Partial<Record<'raster' | 'control' | 'regional_guidance' | 'inpaint_mask', CanvasNodeContract[]>>;
  widgets?: Record<string, Record<string, unknown>>;
  workflowNodes?: unknown[];
} = {}) =>
  ({
    canvas: {
      document: {
        stacks: { control: [], inpaint_mask: [], raster: [], regional_guidance: [], ...stacks },
      },
      // Snapshots are history: a deleted name here must not count as in use.
      snapshots: [{ document: { stacks: { raster: [rasterLayer('old', 'snapshot-only.png')] } } }],
    },
    widgetInstances: Object.fromEntries(
      Object.entries(widgets).map(([typeId, values]) => [typeId, widget(typeId, values)])
    ),
    workflows: {
      activeWorkflowId: 'wf',
      entries: [{ document: { edges: [], name: 'Upscale pass', nodes: workflowNodes } }],
    },
  }) as unknown as Parameters<typeof collectMediaUsage>[0];

const image = (name: string): GalleryItemRef => ({ kind: 'image', name });
const video = (name: string): GalleryItemRef => ({ kind: 'video', name });

describe('collectMediaUsage', () => {
  it('reports nothing for an unused selection or an empty one', () => {
    const doc = project({ stacks: { raster: [rasterLayer('a', 'kept.png')] } });

    expect(collectMediaUsage(doc, [image('other.png')])).toEqual([]);
    expect(collectMediaUsage(doc, [])).toEqual([]);
  });

  it('names each surface that uses a selected image, in panel order, with only the selected names', () => {
    const doc = project({
      stacks: {
        control: [rasterLayer('c', 'control.png')],
        inpaint_mask: [rasterLayer('m', 'mask.png')],
        raster: [rasterLayer('r1', 'raster.png'), rasterLayer('r2', 'raster.png'), rasterLayer('r3', 'other.png')],
        regional_guidance: [
          {
            id: 'rg',
            referenceImages: [{ config: { image: { image_name: 'regional-ref.png' } } }],
            type: 'regional_guidance',
          } as unknown as CanvasNodeContract,
        ],
      },
      widgets: {
        generate: { referenceImages: [{ config: { image: { image_name: 'ref.png' } } }] },
        upscale: { inputImage: { height: 1, image_name: 'upscale.png', width: 1 } },
        video: { firstFrameImage: { height: 1, image_name: 'frame.png', width: 1 } },
      },
      workflowNodes: [{ data: { inputs: { image: { value: { image_name: 'wf.png' } } } } }],
    });
    const selected = [
      image('wf.png'),
      image('frame.png'),
      image('upscale.png'),
      image('regional-ref.png'),
      image('mask.png'),
      image('control.png'),
      image('raster.png'),
      image('ref.png'),
      image('unused.png'),
    ];

    expect(collectMediaUsage(doc, selected)).toEqual([
      { names: ['ref.png'], surface: 'generateReferenceImages' },
      { names: ['raster.png'], surface: 'canvasRaster' },
      { names: ['control.png'], surface: 'canvasControl' },
      { names: ['regional-ref.png'], surface: 'canvasRegionalGuidance' },
      { names: ['mask.png'], surface: 'canvasInpaintMask' },
      { names: ['upscale.png'], surface: 'upscale' },
      { names: ['frame.png'], surface: 'video' },
      { label: 'Upscale pass', names: ['wf.png'], surface: 'workflow' },
    ]);
  });

  it('matches videos by video keys only, so an image and a video sharing a name do not cross over', () => {
    const doc = project({
      widgets: {
        video: {
          references: [{ clip: { video_name: 'same-name' }, kind: 'video' }],
          sourceVideo: { video_name: 'clip.mp4' },
        },
      },
      workflowNodes: [{ data: { inputs: { image: { value: { image_name: 'same-name' } } } } }],
    });

    expect(collectMediaUsage(doc, [video('clip.mp4'), video('same-name'), image('same-name')])).toEqual([
      { names: ['clip.mp4', 'same-name'], surface: 'video' },
      { label: 'Upscale pass', names: ['same-name'], surface: 'workflow' },
    ]);
  });

  it('ignores canvas snapshots and media the gallery merely selects', () => {
    const doc = project({ widgets: { gallery: { selectedImageName: 'snapshot-only.png' } } });

    expect(collectMediaUsage(doc, [image('snapshot-only.png')])).toEqual([]);
  });
});
