import { describe, expect, it } from 'vitest';

import { collectLiveAssetRefs, remapAssetRefs, selectCoverImageName } from './projectAssets';

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
  it('excludes the gallery selection, of either kind', () => {
    const refs = collectLiveAssetRefs(
      projectDocument({
        widgetInstances: {
          'gallery-1': {
            state: {
              values: {
                selectedImage: { fullUrl: '', kind: 'video', name: 'selected.mp4' },
                selectedImageName: 'video:selected.mp4',
                selectedImageNames: ['video:selected.mp4'],
              },
            },
            typeId: 'gallery',
          },
        },
      })
    );

    expect(refs.videos).toEqual(new Set());
    expect(refs.images.has('selected.mp4')).toBe(false);
  });

  it('keeps excluding the gallery selection even if it starts spelling its name `imageName`', () => {
    const { images } = collectLiveAssetRefs(
      projectDocument({
        widgetInstances: {
          'gallery-1': {
            state: { values: { selectedImage: { imageName: 'selected.png', kind: 'image' } } },
            typeId: 'gallery',
          },
        },
      })
    );

    expect(images.has('selected.png')).toBe(false);
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

  it('is null for a project that has produced nothing', () => {
    expect(selectCoverImageName({ canvas: { document: { layers: [] } }, id: 'p', layout: {}, name: 'n' })).toBeNull();
  });

  it('is null for a document missing the canvas entirely', () => {
    expect(selectCoverImageName({ id: 'p', layout: {}, name: 'n' })).toBeNull();
  });
});
