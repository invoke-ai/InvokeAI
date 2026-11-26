import { describe, expect, it, vi } from 'vitest';

import {
  collectHeldAssetRefs,
  collectLiveAssetRefs,
  createLiveCanvasEngines,
  createOpenProjectsHeldMediaReader,
  remapAssetRefs,
  selectCoverImageName,
  stripInstallationState,
} from './projectAssets';

const imageRef = (imageName: string) => ({ height: 512, imageName, width: 512 });

const imageLayer = (id: string, imageName: string) => ({
  id,
  name: id,
  source: { image: imageRef(imageName), type: 'image' },
  type: 'raster',
});

const galleryInstance = (recentImageNames: string[]) => ({
  'gallery-1': {
    state: {
      values: { recentImages: recentImageNames.map((imageName) => ({ ...imageRef(imageName), imageUrl: '' })) },
    },
    typeId: 'gallery',
  },
});

const noMappings = { images: new Map<string, string>(), videos: new Map<string, string>() };

it('holds media from unsaved editor state and undo history', () => {
  const refs = collectHeldAssetRefs([
    {
      canvas: { imageName: 'unsaved.png' },
      queue: { items: [{ imageName: 'completed-job.png' }] },
      widgetInstances: galleryInstance(['gallery-recent.png']),
      undoRedo: { past: [{ project: { imageName: 'undo.png', video_name: 'undo.mp4' } }] },
    },
  ]);

  expect(refs.images).toEqual(new Set(['unsaved.png', 'undo.png']));
  expect(refs.videos).toEqual(new Set(['undo.mp4']));
});

it('holds names Canvas undo retains after the project stops naming them, and drops closed projects', () => {
  let projects = [{ canvas: { imageName: 'second.png' }, id: 'project-1', video: { videoName: 'clip.mp4' } }];
  let undo = ['first.png'];
  const read = createOpenProjectsHeldMediaReader(
    () => projects,
    () => ({ images: undo, videos: [] })
  );

  const held = read();
  expect(new Set(held.images)).toEqual(new Set(['first.png', 'second.png']));
  expect(held.videos).toEqual(['clip.mp4']);

  undo = [];
  projects = [];
  expect(read()).toEqual({ images: [], videos: [] });
});

it('keeps the held result while edits touch only name-free subtrees, and sees names nested edits add', () => {
  const document = { stacks: { raster: [{ source: { image: { imageName: 'layer.png' } } }] } };
  const undoEntry = { project: { widgetInstances: { w: { imageName: 'undo.png' } } } };
  let project = {
    canvas: { document, snapshots: [{ document: { imageName: 'snapshot.png' } }] },
    id: 'project-1',
    layout: { size: 1 },
    undoRedo: { future: [], past: [undoEntry] },
  };
  const read = createOpenProjectsHeldMediaReader(
    () => [project],
    () => undefined
  );
  const first = read();
  expect(new Set(first.images)).toEqual(new Set(['layer.png', 'snapshot.png', 'undo.png']));

  project = { ...project, layout: { size: 2 } };
  expect(read()).toBe(first);
  project = { ...project, canvas: { ...project.canvas, document: { ...document } } };
  const rescanned = read();
  expect(new Set(rescanned.images)).toEqual(new Set(first.images));
  expect(read()).toBe(rescanned);

  project = {
    ...project,
    canvas: {
      ...project.canvas,
      document: { stacks: { raster: [{ source: { image: { imageName: 'stroke.png' } } }] } },
    },
    undoRedo: { future: [], past: [undoEntry, { project: { widgetInstances: { w: { imageName: 'next.png' } } } }] },
  };
  expect(new Set(read().images)).toEqual(new Set(['stroke.png', 'snapshot.png', 'undo.png', 'next.png']));
});

it('announces engines registering and releasing their held media', () => {
  const sources = createLiveCanvasEngines();
  const onChange = vi.fn();
  sources.subscribe(onChange);
  let notifyEngine = () => undefined as void;
  const release = sources.register('project-1', {
    commitFrame: () => false,
    flushPendingPixels: () => Promise.resolve(),
    heldAssets: () => ({ images: ['undo.png'], videos: [] }),
    subscribe: (listener) => {
      notifyEngine = listener;
      return () => undefined;
    },
  });
  expect(sources.heldAssets('project-1')).toEqual({ images: ['undo.png'], videos: [] });
  notifyEngine();
  release();
  expect(sources.heldAssets('project-1')).toBeUndefined();
  expect(onChange).toHaveBeenCalledTimes(3);
});

it("crosses a registered engine's paint barrier and treats an unregistered project as saved", async () => {
  const sources = createLiveCanvasEngines();
  const flushPendingPixels = vi.fn(() => Promise.reject(new Error('upload failed')));
  sources.register('project-1', {
    commitFrame: () => false,
    flushPendingPixels,
    heldAssets: () => ({ images: [], videos: [] }),
    subscribe: () => () => undefined,
  });

  await expect(sources.flushPendingPixels('project-1')).rejects.toThrow('upload failed');
  await expect(sources.flushPendingPixels('project-2')).resolves.toBeUndefined();
  expect(flushPendingPixels).toHaveBeenCalledOnce();
});

const projectDocument = (overrides: Record<string, unknown> = {}): Record<string, unknown> => ({
  canvas: {
    document: { stacks: { raster: [imageLayer('layer-1', 'live-image.png')] }, version: 3 },
    snapshots: [{ document: { stacks: { raster: [imageLayer('layer-old', 'snapshot-image.png')] } }, id: 'snap-1' }],
    version: 3,
  },
  events: [{ imageName: 'event-image.png' }],
  graphHistory: [{ document: { nodes: [{ data: { image: { image_name: 'history-image.png' } } }] }, id: 'gh-1' }],
  id: 'project-1',
  layout: {},
  name: 'Project',
  projectGraph: { nodes: [{ data: { inputs: { image: { image_name: 'graph-image.png' } } }, id: 'node-1' }] },
  queue: {
    items: [
      { id: 'q-1', snapshot: { canvas: { document: { stacks: { raster: [imageLayer('l', 'queue-image.png')] } } } } },
    ],
  },
  widgetInstances: {
    'upscale-1': { state: { values: { inputImage: { image_name: 'upscale-input.png' } } }, typeId: 'upscale' },
  },
  ...overrides,
});

describe('collectLiveAssetRefs', () => {
  it('collects canvas, widget value and graph node references as images', () => {
    const refs = collectLiveAssetRefs(projectDocument());

    expect(refs.images).toEqual(new Set(['live-image.png', 'graph-image.png', 'upscale-input.png']));
    expect(refs.videos).toEqual(new Set());
  });

  it('finds references nested through groups, arrays and mask bitmaps', () => {
    const { images } = collectLiveAssetRefs(
      projectDocument({
        canvas: {
          document: {
            stacks: {
              inpaint_mask: [
                {
                  children: [{ id: 'mask-1', mask: { bitmap: imageRef('mask.png'), fill: {} }, type: 'inpaint_mask' }],
                  id: 'group-1',
                  type: 'group',
                },
              ],
              regional_guidance: [
                {
                  id: 'rg-1',
                  referenceImages: [{ config: { image: imageRef('reference.png'), type: 'ip_adapter' } }],
                  type: 'regional_guidance',
                },
              ],
            },
          },
        },
      })
    );

    expect(images).toEqual(new Set(['mask.png', 'reference.png', 'graph-image.png', 'upscale-input.png']));
  });

  it("holds a control layer's pre-detection original beside its derived map", () => {
    const control = {
      controlType: 'depth',
      detection: { settings: {}, type: 'depth_anything_depth_estimation' },
      id: 'control-1',
      original: { image: imageRef('photo.png'), type: 'image' },
      source: { bitmap: imageRef('depth-map.png'), type: 'paint' },
      type: 'control',
    };
    const document = projectDocument({ canvas: { document: { stacks: { control: [control] } } } });

    expect(collectLiveAssetRefs(document).images).toEqual(
      new Set(['photo.png', 'depth-map.png', 'graph-image.png', 'upscale-input.png'])
    );
    expect(collectHeldAssetRefs([document]).images.has('photo.png')).toBe(true);
  });

  /** Video and image references have separate namespaces, even with identical names. */
  it('collects a graph node video as a video, never as an image', () => {
    const refs = collectLiveAssetRefs(
      projectDocument({
        projectGraph: { nodes: [{ data: { inputs: { video: { video_name: 'clip.mp4' } } }, id: 'extract-1' }] },
      })
    );

    expect(refs.videos).toEqual(new Set(['clip.mp4']));
    expect(refs.images.has('clip.mp4')).toBe(false);
  });
});

/** Strip skipped selection references so unbundled names cannot travel broken and unreported. */
describe('stripInstallationState', () => {
  it('drops every selection key at any depth', () => {
    const stripped = stripInstallationState({
      widgetInstances: {
        'gallery-1': {
          state: {
            values: {
              recentImages: [{ ...imageRef('recent.png'), imageUrl: '' }],
              selectedImage: { imageName: 'selected.png', kind: 'image' },
              selectedImageName: 'image:selected.png',
              selectedImageNames: ['image:selected.png'],
            },
          },
          typeId: 'gallery',
        },
        'preview-1': {
          state: { values: { compareImage: { ...imageRef('compare.png'), imageUrl: '' } } },
          typeId: 'preview',
        },
      },
    });

    expect(JSON.stringify(stripped)).not.toContain('selected.png');
    expect(JSON.stringify(stripped)).not.toContain('compare.png');
    // History is not selection: those references travel deliberately.
    expect(JSON.stringify(stripped)).toContain('recent.png');
  });

  /** Gallery board IDs are installation-specific; hydration replaces them with server-authoritative IDs. */
  it('drops the gallery board ids from both the current and legacy widget shapes', () => {
    const stripped = stripInstallationState({
      widgetInstances: {
        'gallery-1': {
          state: {
            values: {
              autoAddBoardId: 'board-3',
              galleryView: 'images',
              projectBoardId: 'board-1',
              selectedBoardId: 'board-2',
            },
          },
          typeId: 'gallery',
        },
      },
      widgetStates: { gallery: { values: { projectBoardId: 'board-1', selectedBoardId: 'board-2' } } },
    });

    expect(JSON.stringify(stripped)).not.toContain('board-1');
    expect(JSON.stringify(stripped)).not.toContain('board-2');
    expect(JSON.stringify(stripped)).not.toContain('board-3');
    // Everything else the widget holds survives.
    expect(JSON.stringify(stripped)).toContain('galleryView');
  });

  /** Invalidate cached URLs when transfer changes names or server roots. */
  it('blanks the cached media URLs rather than dropping them', () => {
    const stripped = stripInstallationState({
      widgetInstances: {
        'gallery-1': {
          state: {
            values: {
              recentImages: [
                {
                  height: 1,
                  imageName: 'recent.png',
                  imageUrl: 'http://source-install/api/v1/images/i/recent.png/full',
                  queuedAt: '2026-08-07T00:00:00.000Z',
                  sourceQueueItemId: 'q1',
                  thumbnailUrl: 'http://source-install/api/v1/images/i/recent.png/thumbnail',
                  width: 1,
                },
              ],
            },
          },
          typeId: 'gallery',
        },
      },
    });

    const [recent] = (
      (stripped.widgetInstances as Record<string, { state: { values: { recentImages: Record<string, unknown>[] } } }>)[
        'gallery-1'
      ] as { state: { values: { recentImages: Record<string, unknown>[] } } }
    ).state.values.recentImages;

    // Preserve URL keys as blank strings so recents entries survive.
    expect(recent).toMatchObject({ imageName: 'recent.png', imageUrl: '', thumbnailUrl: '' });
    expect(Object.keys(recent!)).toContain('thumbnailUrl');
    expect(JSON.stringify(stripped)).not.toContain('source-install');
  });
});

describe('remapAssetRefs', () => {
  it('rewrites history references too, so nothing keeps pointing at the pre-import name', () => {
    const remapped = remapAssetRefs(projectDocument(), {
      ...noMappings,
      images: new Map([
        ['queue-image.png', 'uploaded-queue.png'],
        ['snapshot-image.png', 'uploaded-snapshot.png'],
      ]),
    }) as {
      canvas: { snapshots: { document: { stacks: { raster: { source: { image: { imageName: string } } }[] } } }[] };
      queue: {
        items: {
          snapshot: { canvas: { document: { stacks: { raster: { source: { image: { imageName: string } } }[] } } } };
        }[];
      };
    };

    expect(remapped.queue.items[0]!.snapshot.canvas.document.stacks.raster[0]!.source.image.imageName).toBe(
      'uploaded-queue.png'
    );
    expect(remapped.canvas.snapshots[0]!.document.stacks.raster[0]!.source.image.imageName).toBe(
      'uploaded-snapshot.png'
    );
  });

  /** The namespaces are separate, so a shared name must not cross over. */
  it('never rewrites an image through the video mapping, or the reverse', () => {
    const document = projectDocument({
      canvas: { document: { stacks: { raster: [imageLayer('l', 'shared')] } } },
      projectGraph: { nodes: [{ data: { inputs: { video: { video_name: 'shared' } } }, id: 'n' }] },
    });
    const remapped = remapAssetRefs(document, {
      images: new Map([['shared', 'image-side']]),
      videos: new Map([['shared', 'video-side']]),
    }) as {
      canvas: { document: { stacks: { raster: { source: { image: { imageName: string } } }[] } } };
      projectGraph: { nodes: { data: { inputs: { video: { video_name: string } } } }[] };
    };

    expect(remapped.canvas.document.stacks.raster[0]!.source.image.imageName).toBe('image-side');
    expect(remapped.projectGraph.nodes[0]!.data.inputs.video.video_name).toBe('video-side');
  });
});

describe('selectCoverImageName', () => {
  it('reads the top-most raster leaf through nested groups', () => {
    const document = projectDocument({
      canvas: {
        document: {
          stacks: {
            raster: [
              {
                children: [
                  {
                    children: [{ id: 'deep', source: { image: imageRef('deep.png'), type: 'image' }, type: 'raster' }],
                    id: 'inner',
                    type: 'group',
                  },
                  { id: 'later', source: { image: imageRef('later.png'), type: 'image' }, type: 'raster' },
                ],
                id: 'outer',
                type: 'group',
              },
            ],
          },
        },
      },
    });
    expect(selectCoverImageName(document)).toBe('deep.png');
  });
});
