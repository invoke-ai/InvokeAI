import { describe, expect, it } from 'vitest';

import { collectLiveImageRefs, remapImageRefs, selectCoverImageName } from './projectAssets';

/**
 * The walker's contract, stated as documents rather than as paths: what an
 * archive must bundle, what it must leave behind, and what has to be rewritten
 * when the server hands back a different name.
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

describe('collectLiveImageRefs', () => {
  it('collects canvas, widget value and graph node references', () => {
    expect(collectLiveImageRefs(projectDocument())).toEqual(
      new Set(['live-image.png', 'graph-image.png', 'upscale-input.png'])
    );
  });

  it('leaves history behind — queue snapshots, graph history, events, canvas snapshots', () => {
    const collected = collectLiveImageRefs(projectDocument());

    expect(collected.has('queue-image.png')).toBe(false);
    expect(collected.has('history-image.png')).toBe(false);
    expect(collected.has('event-image.png')).toBe(false);
    expect(collected.has('snapshot-image.png')).toBe(false);
  });

  it('leaves the gallery widget recents behind', () => {
    const collected = collectLiveImageRefs(
      projectDocument({ widgetInstances: galleryInstance(['recent-a.png', 'recent-b.png']) })
    );

    expect(collected.has('recent-a.png')).toBe(false);
  });

  it('finds references nested through arrays and mask bitmaps', () => {
    const collected = collectLiveImageRefs(
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

    expect(collected).toEqual(new Set(['mask.png', 'reference.png', 'graph-image.png', 'upscale-input.png']));
  });

  it('ignores empty names and non-string values', () => {
    expect(collectLiveImageRefs({ canvas: { imageName: '' }, settings: { image_name: 7 } })).toEqual(new Set());
  });
});

describe('remapImageRefs', () => {
  it('returns the document unchanged for an empty mapping', () => {
    const document = projectDocument();

    expect(remapImageRefs(document, new Map())).toBe(document);
  });

  it('rewrites live references', () => {
    const remapped = remapImageRefs(projectDocument(), new Map([['live-image.png', 'uploaded-1.png']])) as {
      canvas: { document: { layers: { source: { image: { imageName: string } } }[] } };
    };

    expect(remapped.canvas.document.layers[0]!.source.image.imageName).toBe('uploaded-1.png');
  });

  it('rewrites history references too, so nothing keeps pointing at the pre-import name', () => {
    const remapped = remapImageRefs(
      projectDocument(),
      new Map([
        ['queue-image.png', 'uploaded-queue.png'],
        ['snapshot-image.png', 'uploaded-snapshot.png'],
      ])
    ) as {
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
    const remapped = remapImageRefs(projectDocument(), new Map([['graph-image.png', 'uploaded-graph.png']])) as {
      projectGraph: { nodes: { data: { inputs: { image: { image_name: string } } } }[] };
    };

    expect(remapped.projectGraph.nodes[0]!.data.inputs.image.image_name).toBe('uploaded-graph.png');
  });

  it('leaves names absent from the mapping alone', () => {
    const remapped = remapImageRefs(projectDocument(), new Map([['not-present.png', 'other.png']])) as {
      canvas: { document: { layers: { source: { image: { imageName: string } } }[] } };
    };

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
