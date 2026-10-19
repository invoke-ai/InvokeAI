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

const paintLayer = (id: string, imageName: string | null) => ({
  id,
  name: id,
  source: { bitmap: imageName ? imageRef(imageName) : null, type: 'paint' },
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

it('rescans only the workflow an edit touched, sharing the documents its history retains', () => {
  const docA = { id: 'a', nodes: [{ data: { inputs: { image: { value: { image_name: 'a.png' } } } } }] };
  const docB = { id: 'b', nodes: [{ data: { inputs: { image: { value: { image_name: 'b.png' } } } } }] };
  let project = {
    id: 'project-1',
    workflowHistories: {},
    workflows: { activeWorkflowId: 'a', entries: [{ document: docA }, { document: docB }] },
  };
  const read = createOpenProjectsHeldMediaReader(
    () => [project],
    () => undefined
  );
  const first = read();
  expect(new Set(first.images)).toEqual(new Set(['a.png', 'b.png']));

  // Editing `a` keeps `docA` in a's history; b's entry is untouched and its scan is reused by identity.
  const docA2 = { ...docA, nodes: [{ data: { inputs: { image: { value: { image_name: 'a2.png' } } } } }] };
  project = {
    ...project,
    workflowHistories: { a: { future: [], past: [{ document: docA, id: 'h1', sequence: 1 }] } },
    workflows: { ...project.workflows, entries: [{ document: docA2 }, project.workflows.entries[1]!] },
  };
  expect(new Set(read().images)).toEqual(new Set(['a.png', 'a2.png', 'b.png']));

  // Removing `a` releases its history: nothing holds a.png or a2.png any more.
  project = {
    ...project,
    workflowHistories: {},
    workflows: { activeWorkflowId: 'b', entries: [project.workflows.entries[1]!] },
  };
  expect(read().images).toEqual(['b.png']);
});

it('announces engines registering and releasing their held media', () => {
  const sources = createLiveCanvasEngines();
  const onChange = vi.fn();
  sources.subscribe(onChange);
  let notifyEngine = () => undefined as void;
  const release = sources.register('project-1', {
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

  it('leaves history behind — queue snapshots, graph history, events, canvas snapshots', () => {
    const { images } = collectLiveAssetRefs(projectDocument());

    expect(images.has('queue-image.png')).toBe(false);
    expect(images.has('history-image.png')).toBe(false);
    expect(images.has('event-image.png')).toBe(false);
    expect(images.has('snapshot-image.png')).toBe(false);
  });

  it('leaves the gallery widget recents behind', () => {
    const { images } = collectLiveAssetRefs(
      projectDocument({ widgetInstances: galleryInstance(['recent-a.png', 'recent-b.png']) })
    );

    expect(images.has('recent-a.png')).toBe(false);
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

  it('ignores empty names and non-string values', () => {
    const refs = collectLiveAssetRefs({ canvas: { imageName: '' }, settings: { image_name: 7, video_name: null } });

    expect(refs.images).toEqual(new Set());
    expect(refs.videos).toEqual(new Set());
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

  it('returns both sets for a document carrying both kinds', () => {
    const refs = collectLiveAssetRefs(
      projectDocument({
        projectGraph: {
          nodes: [
            { data: { inputs: { image: { image_name: 'frame.png' }, video: { video_name: 'clip.mp4' } } }, id: 'n' },
          ],
        },
      })
    );

    expect(refs.images).toEqual(new Set(['live-image.png', 'frame.png', 'upscale-input.png']));
    expect(refs.videos).toEqual(new Set(['clip.mp4']));
  });

  /** Exclude selection by parent key regardless of media-name spelling. */
  /** Exclude compare/selection references independently of recognized imageName keys. */
  it.each([
    [
      'the selection, of either kind',
      {
        selectedImage: { fullUrl: '', kind: 'video', name: 'selected.mp4' },
        selectedImageName: 'video:selected.mp4',
        selectedImageNames: ['video:selected.mp4'],
      },
      'selected.mp4',
    ],
    ['the compare selection', { compareImage: { ...imageRef('compare.png'), imageUrl: '' } }, 'compare.png'],
    [
      'a selection spelling its name `imageName`',
      { selectedImage: { imageName: 'selected.png', kind: 'image' } },
      'selected.png',
    ],
  ])('excludes %s', (_label, values, excluded) => {
    const refs = collectLiveAssetRefs(
      projectDocument({ widgetInstances: { 'gallery-1': { state: { values }, typeId: 'gallery' } } })
    );

    expect(refs.images.has(excluded)).toBe(false);
    expect(refs.videos.has(excluded)).toBe(false);
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

  it('keeps the surrounding widget state intact', () => {
    const stripped = stripInstallationState({
      widgetInstances: {
        'gallery-1': { state: { values: { boardId: 'b1', selectedImageName: 'image:x.png' } }, typeId: 'gallery' },
      },
    });

    expect(stripped).toEqual({
      widgetInstances: { 'gallery-1': { state: { values: { boardId: 'b1' } }, typeId: 'gallery' } },
    });
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

  /** Preserve unchanged identity and authored workflow board inputs. */
  it.each([
    ['no selection', { canvas: { document: { stacks: { raster: [imageLayer('l1', 'a.png')] } } }, id: 'p1' }],
    [
      'an already-blank URL',
      { widgetStates: { gallery: { values: { recentImages: [{ imageName: 'a.png', imageUrl: '' }] } } } },
    ],
    [
      'a board reference that means something',
      { workflow: { nodes: [{ inputs: { board: { value: { board_id: 'board-3' } } }, type: 'save_image' }] } },
    ],
  ])('returns a document with %s unchanged', (_label, document) => {
    expect(stripInstallationState(document)).toBe(document);
  });
});

describe('remapAssetRefs', () => {
  it('returns the document unchanged for empty mappings', () => {
    const document = projectDocument();

    expect(remapAssetRefs(document, noMappings)).toBe(document);
  });

  it('rewrites live references', () => {
    const remapped = remapAssetRefs(projectDocument(), {
      ...noMappings,
      images: new Map([['live-image.png', 'uploaded-1.png']]),
    }) as { canvas: { document: { stacks: { raster: { source: { image: { imageName: string } } }[] } } } };

    expect(remapped.canvas.document.stacks.raster[0]!.source.image.imageName).toBe('uploaded-1.png');
  });

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

  it('rewrites the backend spelling of the key as well', () => {
    const remapped = remapAssetRefs(projectDocument(), {
      ...noMappings,
      images: new Map([['graph-image.png', 'uploaded-graph.png']]),
    }) as { projectGraph: { nodes: { data: { inputs: { image: { image_name: string } } } }[] } };

    expect(remapped.projectGraph.nodes[0]!.data.inputs.image.image_name).toBe('uploaded-graph.png');
  });

  it('rewrites a video through the video mapping', () => {
    const document = projectDocument({
      projectGraph: { nodes: [{ data: { inputs: { video: { video_name: 'clip.mp4' } } }, id: 'n' }] },
    });
    const remapped = remapAssetRefs(document, {
      ...noMappings,
      videos: new Map([['clip.mp4', 'server-clip.mp4']]),
    }) as { projectGraph: { nodes: { data: { inputs: { video: { video_name: string } } } }[] } };

    expect(remapped.projectGraph.nodes[0]!.data.inputs.video.video_name).toBe('server-clip.mp4');
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

  it('leaves names absent from the mapping alone', () => {
    const remapped = remapAssetRefs(projectDocument(), {
      ...noMappings,
      images: new Map([['not-present.png', 'other.png']]),
    }) as { canvas: { document: { stacks: { raster: { source: { image: { imageName: string } } }[] } } } };

    expect(remapped.canvas.document.stacks.raster[0]!.source.image.imageName).toBe('live-image.png');
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

  it('prefers the newest gallery result', () => {
    expect(
      selectCoverImageName(projectDocument({ widgetInstances: galleryInstance(['newest.png', 'older.png']) }))
    ).toBe('newest.png');
  });

  it('falls back to the top-most canvas layer with pixels', () => {
    expect(selectCoverImageName(projectDocument())).toBe('live-image.png');
  });

  it('skips canvas layers that have no pixels yet', () => {
    expect(
      selectCoverImageName(
        projectDocument({
          canvas: { document: { stacks: { raster: [paintLayer('empty', null), imageLayer('below', 'below.png')] } } },
        })
      )
    ).toBe('below.png');
  });

  it.each([
    [
      'a project that has produced nothing',
      { canvas: { document: { stacks: { raster: [] } } }, id: 'p', layout: {}, name: 'n' },
    ],
    ['a document missing the canvas entirely', { id: 'p', layout: {}, name: 'n' }],
  ])('is null for %s', (_label, document) => {
    expect(selectCoverImageName(document)).toBeNull();
  });
});
