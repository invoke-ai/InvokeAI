import { describe, expect, it } from 'vitest';

import { collectLiveAssetRefs, remapAssetRefs, selectCoverImageName, stripInstallationState } from './projectAssets';

/**
 * The walker's contract, stated as documents rather than as paths: what an
 * archive must bundle, what it must leave behind, which kind each name is, and
 * what has to be rewritten when the server hands back a different name.
 */

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

const projectDocument = (overrides: Record<string, unknown> = {}): Record<string, unknown> => ({
  canvas: {
    document: { layers: [imageLayer('layer-1', 'live-image.png')], version: 2 },
    snapshots: [{ document: { layers: [imageLayer('layer-old', 'snapshot-image.png')] }, id: 'snap-1' }],
    version: 2,
  },
  events: [{ imageName: 'event-image.png' }],
  graphHistory: [{ document: { nodes: [{ data: { image: { image_name: 'history-image.png' } } }] }, id: 'gh-1' }],
  id: 'project-1',
  layout: {},
  name: 'Project',
  projectGraph: { nodes: [{ data: { inputs: { image: { image_name: 'graph-image.png' } } }, id: 'node-1' }] },
  queue: {
    items: [{ id: 'q-1', snapshot: { canvas: { document: { layers: [imageLayer('l', 'queue-image.png')] } } } }],
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

  it('finds references nested through arrays and mask bitmaps', () => {
    const { images } = collectLiveAssetRefs(
      projectDocument({
        canvas: {
          document: {
            layers: [
              { id: 'mask-1', mask: { bitmap: imageRef('mask.png'), fill: {} }, type: 'inpaint_mask' },
              {
                id: 'rg-1',
                referenceImages: [{ config: { image: imageRef('reference.png'), type: 'ip_adapter' } }],
                type: 'regional_guidance',
              },
            ],
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

  /**
   * `VideoField { video_name }` reaches `projectGraph` through an imported
   * workflow. It is a separate namespace from images and must not be conflated.
   */
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

  /**
   * The gallery selection is a pointer into per-install gallery content, not
   * something the document renders, so neither kind is bundled. The second
   * assertion is the regression guard: the exclusion is by key, so it survives
   * `GalleryItem` spelling its name field the same way the canvas does.
   */
  /**
   * `compareImage` is a `GeneratedImageContract`, so it carries `imageName` and would otherwise be
   * collected. A comparison left open is not project content. The last row guards the exclusion
   * against `selectedImage` starting to spell its name `imageName`, which no collector key matches
   * today only by accident.
   */
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

/**
 * Not bundling the selection is only half of leaving it behind. A reference
 * that is skipped by the collector but left in the document still travels —
 * broken, and invisible to the restore pass that would otherwise report it as
 * dangling.
 */
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

  /**
   * A board id means nothing on the machine a project arrives at, and the one
   * naming the project's own board is a cache the server overwrites anyway.
   */
  it('drops the gallery board ids from both the current and legacy widget shapes', () => {
    const stripped = stripInstallationState({
      widgetInstances: {
        'gallery-1': {
          state: { values: { galleryView: 'images', projectBoardId: 'board-1', selectedBoardId: 'board-2' } },
          typeId: 'gallery',
        },
      },
      widgetStates: { gallery: { values: { projectBoardId: 'board-1', selectedBoardId: 'board-2' } } },
    });

    expect(JSON.stringify(stripped)).not.toContain('board-1');
    expect(JSON.stringify(stripped)).not.toContain('board-2');
    // Everything else the widget holds survives.
    expect(JSON.stringify(stripped)).toContain('galleryView');
  });

  /**
   * A URL is built around a media name and a server. The name is remapped by a transfer; the URL is
   * not — so a cached one keeps resolving, to the *source* project's picture, on a document that no
   * longer references that media at all.
   */
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

    // Blanked, not removed: `getBoundedRecentImages` requires both to be strings and silently drops
    // any entry missing one, so deleting the keys would discard the whole recents overlay.
    expect(recent).toMatchObject({ imageName: 'recent.png', imageUrl: '', thumbnailUrl: '' });
    expect(Object.keys(recent!)).toContain('thumbnailUrl');
    expect(JSON.stringify(stripped)).not.toContain('source-install');
  });

  /**
   * Subtrees with nothing to change keep their identity. The last row matters most: only the
   * gallery widget's own two keys are installation state, so a board named by a workflow node is an
   * authored input that has to survive the round trip.
   */
  it.each([
    ['no selection', { canvas: { document: { layers: [imageLayer('l1', 'a.png')] } }, id: 'p1' }],
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
    }) as { canvas: { document: { layers: { source: { image: { imageName: string } } }[] } } };

    expect(remapped.canvas.document.layers[0]!.source.image.imageName).toBe('uploaded-1.png');
  });

  it('rewrites history references too, so nothing keeps pointing at the pre-import name', () => {
    const remapped = remapAssetRefs(projectDocument(), {
      ...noMappings,
      images: new Map([
        ['queue-image.png', 'uploaded-queue.png'],
        ['snapshot-image.png', 'uploaded-snapshot.png'],
      ]),
    }) as {
      canvas: { snapshots: { document: { layers: { source: { image: { imageName: string } } }[] } }[] };
      queue: {
        items: { snapshot: { canvas: { document: { layers: { source: { image: { imageName: string } } }[] } } } }[];
      };
    };

    expect(remapped.queue.items[0]!.snapshot.canvas.document.layers[0]!.source.image.imageName).toBe(
      'uploaded-queue.png'
    );
    expect(remapped.canvas.snapshots[0]!.document.layers[0]!.source.image.imageName).toBe('uploaded-snapshot.png');
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
      canvas: { document: { layers: [imageLayer('l', 'shared')] } },
      projectGraph: { nodes: [{ data: { inputs: { video: { video_name: 'shared' } } }, id: 'n' }] },
    });
    const remapped = remapAssetRefs(document, {
      images: new Map([['shared', 'image-side']]),
      videos: new Map([['shared', 'video-side']]),
    }) as {
      canvas: { document: { layers: { source: { image: { imageName: string } } }[] } };
      projectGraph: { nodes: { data: { inputs: { video: { video_name: string } } } }[] };
    };

    expect(remapped.canvas.document.layers[0]!.source.image.imageName).toBe('image-side');
    expect(remapped.projectGraph.nodes[0]!.data.inputs.video.video_name).toBe('video-side');
  });

  it('leaves names absent from the mapping alone', () => {
    const remapped = remapAssetRefs(projectDocument(), {
      ...noMappings,
      images: new Map([['not-present.png', 'other.png']]),
    }) as { canvas: { document: { layers: { source: { image: { imageName: string } } }[] } } };

    expect(remapped.canvas.document.layers[0]!.source.image.imageName).toBe('live-image.png');
  });
});

describe('selectCoverImageName', () => {
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
          canvas: { document: { layers: [paintLayer('empty', null), imageLayer('below', 'below.png')] } },
        })
      )
    ).toBe('below.png');
  });

  it.each([
    ['a project that has produced nothing', { canvas: { document: { layers: [] } }, id: 'p', layout: {}, name: 'n' }],
    ['a document missing the canvas entirely', { id: 'p', layout: {}, name: 'n' }],
  ])('is null for %s', (_label, document) => {
    expect(selectCoverImageName(document)).toBeNull();
  });
});
