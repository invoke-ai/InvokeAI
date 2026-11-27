import type { GalleryItemRef } from '@features/gallery/contracts';
import type { CanvasNodeContract } from '@workbench/canvas-engine/contracts';

import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { describe, expect, it } from 'vitest';

import { collectMediaUsage, mergeMediaUsage } from './mediaUsage';

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
  fields = {},
  snapshots = [],
  stacks = {},
  staged = [],
  widgets = {},
  workflow = {},
  workflowNodes = [],
}: {
  /** Other project fields, at the root. */
  fields?: Record<string, unknown>;
  snapshots?: unknown[];
  stacks?: Partial<Record<'raster' | 'control' | 'regional_guidance' | 'inpaint_mask', CanvasNodeContract[]>>;
  staged?: unknown[];
  widgets?: Record<string, Record<string, unknown>>;
  /** Other fields of the workflow entry, beside its document. */
  workflow?: Record<string, unknown>;
  workflowNodes?: unknown[];
} = {}) =>
  ({
    canvas: {
      document: {
        stacks: { control: [], inpaint_mask: [], raster: [], regional_guidance: [], ...stacks },
      },
      snapshots,
      stagingArea: { pendingImages: staged },
    },
    widgetInstances: Object.fromEntries(
      Object.entries(widgets).map(([typeId, values]) => [typeId, widget(typeId, values)])
    ),
    workflows: {
      activeWorkflowId: 'wf',
      entries: [{ document: { edges: [], name: 'Upscale pass', nodes: workflowNodes }, ...workflow }],
    },
    ...fields,
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
      { refs: [image('ref.png')], surface: 'generateReferenceImages' },
      { refs: [image('raster.png')], surface: 'canvasRaster' },
      { refs: [image('control.png')], surface: 'canvasControl' },
      { refs: [image('regional-ref.png')], surface: 'canvasRegionalGuidance' },
      { refs: [image('mask.png')], surface: 'canvasInpaintMask' },
      { refs: [image('upscale.png')], surface: 'upscale' },
      { refs: [image('frame.png')], surface: 'video' },
      { label: 'Upscale pass', refs: [image('wf.png')], surface: 'workflow' },
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
      { refs: [video('clip.mp4'), video('same-name')], surface: 'video' },
      { label: 'Upscale pass', refs: [image('same-name')], surface: 'workflow' },
    ]);
  });

  it('counts staged results and each canvas snapshot, the live document and its staged results alike', () => {
    const doc = project({
      snapshots: [
        { document: { stacks: { raster: [rasterLayer('old', 'cleared.png')] } }, name: 'Before clear' },
        {
          document: { stacks: { raster: [] } },
          name: 'Discarded results',
          stagedImages: [{ imageName: 'discarded.png', placement: {} }],
        },
      ],
      staged: [{ imageName: 'staged.png', placement: {} }],
    });

    expect(collectMediaUsage(doc, [image('cleared.png'), image('discarded.png'), image('staged.png')])).toEqual([
      { refs: [image('staged.png')], surface: 'canvasStaging' },
      { label: 'Before clear', refs: [image('cleared.png')], surface: 'canvasSnapshot' },
      { label: 'Discarded results', refs: [image('discarded.png')], surface: 'canvasSnapshot' },
    ]);
  });

  it('ignores what the gallery merely shows: its selection, recents and compare image', () => {
    const doc = project({
      widgets: {
        gallery: {
          compareImageName: 'shown.png',
          recentImages: [{ imageName: 'shown.png' }],
          selectedImageName: 'shown.png',
        },
      },
    });

    expect(collectMediaUsage(doc, [image('shown.png')])).toEqual([]);
  });

  it('reports a name anywhere else in the project as used elsewhere, once, so a field with no surface still warns', () => {
    const doc = project({
      fields: { futureField: { nested: [{ image_name: 'future.png' }] } },
      widgets: {
        // A widget the panel has no surface for, and a Generate value beside its reference images.
        'image-notes': { pinned: { imageName: 'pinned.png' } },
        generate: {
          referenceImages: [{ config: { image: { image_name: 'ref.png' } } }],
          recalled: { image_name: 'ref.png' },
        },
      },
      workflow: { lastRun: { imageName: 'last-run.png' } },
      workflowNodes: [],
    });

    expect(
      collectMediaUsage(doc, [image('pinned.png'), image('ref.png'), image('future.png'), image('last-run.png')])
    ).toEqual([
      { refs: [image('ref.png')], surface: 'generateReferenceImages' },
      { refs: [image('pinned.png'), image('future.png')], surface: 'otherProjectState' },
    ]);

    // An image sharing a video's name is its own item: the video surface does not account for it.
    const sameName = project({
      widgets: {
        'image-notes': { pinned: { imageName: 'same-name' } },
        video: { sourceVideo: { video_name: 'same-name' } },
      },
    });

    expect(collectMediaUsage(sameName, [video('same-name'), image('same-name')])).toEqual([
      { refs: [video('same-name')], surface: 'video' },
      { refs: [image('same-name')], surface: 'otherProjectState' },
    ]);
  });

  it('still reports a canvas stack kind or snapshot field that no surface reads', () => {
    const doc = project({
      snapshots: [{ document: { stacks: {} }, name: 'Before clear', preview: { imageName: 'snapshot-preview.png' } }],
      stacks: { future_stack: [rasterLayer('f', 'future-stack.png')] } as Record<string, CanvasNodeContract[]>,
    });

    expect(collectMediaUsage(doc, [image('future-stack.png'), image('snapshot-preview.png')])).toEqual([
      { refs: [image('future-stack.png'), image('snapshot-preview.png')], surface: 'otherProjectState' },
    ]);
  });

  it('leaves out what only shows or records media, and session state that is never saved', () => {
    const shown = { imageName: 'shown.png' };
    const doc = project({
      fields: {
        events: [{ payload: shown }],
        queue: { items: [{ results: [shown] }] },
        undoRedo: { future: [], past: [{ project: { widgetInstances: { a: shown } } }] },
        widgetGraphs: { generate: { nodes: [{ inputs: { image: { image_name: 'shown.png' } } }] } },
        workflowHistories: { wf: { future: [], past: [{ document: { nodes: [shown] } }] } },
      },
      widgets: { gallery: { semanticImageQuery: { imageName: 'shown.png', kind: 'image' } } },
      workflow: { lastRun: shown },
    });

    expect(collectMediaUsage(doc, [image('shown.png')])).toEqual([]);
  });

  it('finds no use in a fresh project that just generated the image and shows it everywhere it records runs', () => {
    const initial = createInitialWorkbenchState();
    const activeProjectId = initial.activeProjectId;
    const atGallery = {
      ...initial,
      projects: initial.projects.map((candidate) =>
        candidate.id === activeProjectId
          ? { ...candidate, invocation: { ...candidate.invocation, destination: 'gallery' as const } }
          : candidate
      ),
    };
    const generated = {
      height: 768,
      imageName: 'generated.png',
      imageUrl: '/api/v1/images/i/generated.png/full',
      queuedAt: '2026-11-03T00:00:00.000Z',
      sourceQueueItemId: 'run-1',
      thumbnailUrl: '/api/v1/images/i/generated.png/thumbnail',
      width: 512,
    };
    // The real result routing: the Gallery's recents and selection, as a finished run leaves them.
    const routed = workbenchReducer(atGallery, {
      images: [generated],
      projectId: activeProjectId,
      queueItemId: 'run-1',
      type: 'routeQueueItemResults',
    });
    const project = routed.projects.find((candidate) => candidate.id === activeProjectId)!;
    // What a run records beside that: its queue entry and results, the event log, the submitted graph, and a workflow
    // run preview.
    const ran = {
      ...project,
      events: [...project.events, { createdAt: generated.queuedAt, id: 'e', image: generated, type: 'run-completed' }],
      invocation: { ...project.invocation, lastSubmittedRunId: 'run-1' },
      queue: { items: [{ id: 'run-1', resultImages: [generated], status: 'completed' }] },
      widgetGraphs: {
        generate: {
          edges: [],
          id: 'g',
          label: 'Generate',
          nodes: [{ id: 'n', inputs: { image: { image_name: 'generated.png' } }, type: 'i2l' }],
          updatedAt: '',
          version: 1,
        },
      },
      workflows: {
        ...project.workflows,
        entries: project.workflows.entries.map((entry) => ({
          ...entry,
          lastRun: { completedAt: '', imageName: 'generated.png', submittedAt: '' },
        })),
      },
    } as unknown as typeof project;

    expect(JSON.stringify(ran.widgetInstances)).toContain('generated.png');
    expect(collectMediaUsage(ran, [image('generated.png')])).toEqual([]);
  });
});

describe('mergeMediaUsage', () => {
  it('joins two copies of a project by surface, workflows and snapshots by id, in panel order', () => {
    const merged = mergeMediaUsage(
      [
        { refs: [image('a.png')], surface: 'canvasRaster' },
        { label: 'Same name', refs: [image('b.png')], sourceId: 'wf-1', surface: 'workflow' },
      ],
      [
        { label: 'Same name', refs: [image('b.png')], sourceId: 'wf-2', surface: 'workflow' },
        // wf-1, renamed in the other copy.
        { label: 'Renamed', refs: [image('d.png')], sourceId: 'wf-1', surface: 'workflow' },
        { refs: [image('c.png')], surface: 'canvasRaster' },
        { refs: [image('f.png')], surface: 'generateReferenceImages' },
        // a.png is on a layer in the other copy: listed there, not also elsewhere.
        { refs: [image('a.png'), image('e.png')], surface: 'otherProjectState' },
      ]
    );

    expect(merged).toEqual([
      { refs: [image('f.png')], surface: 'generateReferenceImages' },
      { refs: [image('a.png'), image('c.png')], surface: 'canvasRaster' },
      { label: 'Same name', refs: [image('b.png'), image('d.png')], sourceId: 'wf-1', surface: 'workflow' },
      { label: 'Same name', refs: [image('b.png')], sourceId: 'wf-2', surface: 'workflow' },
      { refs: [image('e.png')], surface: 'otherProjectState' },
    ]);
  });
});
