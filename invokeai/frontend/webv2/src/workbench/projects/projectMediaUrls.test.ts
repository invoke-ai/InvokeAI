import type { GeneratedImageContract } from '@features/gallery';
import type { CanvasRegionalGuidanceLayerContract } from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';

import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createRegionalGuidanceLayerWithRefImage } from '@workbench/widgets/layers/layerOps';
import { createDraftProject } from '@workbench/workbenchState';
import { describe, expect, it } from 'vitest';

import { rebuildProjectMediaUrls, type ProjectMediaUrlResolver } from './projectMediaUrls';

const urls: ProjectMediaUrlResolver = {
  image: {
    full: (name) => `https://here/images/${name}/full`,
    thumbnail: (name) => `https://here/images/${name}/thumb`,
  },
  video: {
    full: (name) => `https://here/videos/${name}/full`,
    thumbnail: (name) => `https://here/videos/${name}/thumb`,
  },
};

const STALE = 'http://localhost:9090/api/v1';

const generated = (imageName: string, url = `${STALE}/images/i/${imageName}/full`): GeneratedImageContract => ({
  height: 64,
  imageName,
  imageUrl: url,
  queuedAt: '2026-10-19T00:00:00.000Z',
  sourceQueueItemId: 'queue-1',
  thumbnailUrl: url === '' ? '' : `${STALE}/images/i/${imageName}/thumbnail`,
  width: 64,
});

const galleryItem = (kind: 'image' | 'video', name: string) => ({
  boardId: 'none',
  category: 'general',
  createdAt: '2026-10-19T00:00:00.000Z',
  fullUrl: `${STALE}/${kind}s/i/${name}/full`,
  height: 64,
  isIntermediate: false,
  kind,
  name,
  starred: false,
  thumbnailUrl: `${STALE}/${kind}s/i/${name}/thumbnail`,
  width: 64,
  ...(kind === 'video' ? { durationSeconds: 2 } : {}),
});

const regionalWith = (id: string, image: GeneratedImageContract | null): CanvasRegionalGuidanceLayerContract => {
  const layer = createRegionalGuidanceLayerWithRefImage(id, 0, 'sdxl', id);
  const [reference] = layer.referenceImages;
  return { ...layer, referenceImages: [{ ...reference!, config: { ...reference!.config, image } }] };
};

const withGallery = (project: Project, instanceId: string, values: Record<string, unknown>): Project => ({
  ...project,
  widgetInstances: {
    ...project.widgetInstances,
    [instanceId]: { id: instanceId, state: { values }, typeId: 'gallery' } as Project['widgetInstances'][string],
  },
});

/** A project holding a stale or blank URL in every place a project stores one. */
const staleProject = (): Project => {
  const base = createDraftProject([]);
  const document = {
    ...base.canvas.document,
    stacks: stacksFrom([groupContract('group', [regionalWith('nested', generated('nested.png'))])]),
  };
  const withCanvas: Project = {
    ...base,
    canvas: {
      ...base.canvas,
      document,
      snapshots: [
        {
          createdAt: '2026-10-19T00:00:00.000Z',
          document: { ...document, stacks: stacksFrom([regionalWith('snap', generated('snap.png', ''))]) },
          id: 'snapshot-1',
          name: 'Snapshot',
        },
      ],
      stagingArea: {
        ...base.canvas.stagingArea,
        pendingImages: [{ ...generated('staged.png'), placement: { height: 64, opacity: 1, width: 64, x: 0, y: 0 } }],
      },
    },
  };
  return withGallery(
    withGallery(withCanvas, 'gallery', {
      compareImage: generated('compare.png'),
      recentImages: [generated('recent.png'), generated('imported.png', '')],
      selectedImage: galleryItem('image', 'selected.png'),
    }),
    'gallery-2',
    { selectedImage: galleryItem('video', 'clip.mp4') }
  );
};

const galleryValues = (project: Project, instanceId: string) => project.widgetInstances[instanceId]!.state.values;

describe('rebuildProjectMediaUrls', () => {
  it('rebuilds every stored media URL from the name beside it', () => {
    const rebuilt = rebuildProjectMediaUrls(staleProject(), urls);

    expect(galleryValues(rebuilt, 'gallery')).toMatchObject({
      compareImage: {
        imageUrl: 'https://here/images/compare.png/full',
        thumbnailUrl: 'https://here/images/compare.png/thumb',
      },
      recentImages: [
        { imageUrl: 'https://here/images/recent.png/full', thumbnailUrl: 'https://here/images/recent.png/thumb' },
        // `.invk` transfer blanks URLs; a load fills them.
        { imageUrl: 'https://here/images/imported.png/full', thumbnailUrl: 'https://here/images/imported.png/thumb' },
      ],
      selectedImage: {
        fullUrl: 'https://here/images/selected.png/full',
        thumbnailUrl: 'https://here/images/selected.png/thumb',
      },
    });
    expect(galleryValues(rebuilt, 'gallery-2').selectedImage).toMatchObject({
      fullUrl: 'https://here/videos/clip.mp4/full',
      thumbnailUrl: 'https://here/videos/clip.mp4/thumb',
    });
    expect(rebuilt.canvas.stagingArea.pendingImages[0]).toMatchObject({
      imageUrl: 'https://here/images/staged.png/full',
      placement: { height: 64, opacity: 1, width: 64, x: 0, y: 0 },
      thumbnailUrl: 'https://here/images/staged.png/thumb',
    });
    const [group] = rebuilt.canvas.document.stacks.regional_guidance;
    const nested = group?.type === 'group' ? group.children[0] : null;
    expect(nested?.type === 'regional_guidance' ? nested.referenceImages[0]?.config.image : null).toMatchObject({
      imageUrl: 'https://here/images/nested.png/full',
      thumbnailUrl: 'https://here/images/nested.png/thumb',
    });
    const [snapshotLayer] = rebuilt.canvas.snapshots[0]!.document.stacks.regional_guidance;
    expect(
      snapshotLayer?.type === 'regional_guidance' ? snapshotLayer.referenceImages[0]?.config.image : null
    ).toMatchObject({
      imageUrl: 'https://here/images/snap.png/full',
      thumbnailUrl: 'https://here/images/snap.png/thumb',
    });
    expect(JSON.stringify(rebuilt)).not.toContain('localhost:9090');
  });

  it('keeps the identity of a project, and of every untouched subtree, whose URLs already match', () => {
    const rebuilt = rebuildProjectMediaUrls(staleProject(), urls);

    expect(rebuildProjectMediaUrls(rebuilt, urls)).toBe(rebuilt);

    const base = createDraftProject([]);
    const selectedOnly = withGallery(base, 'gallery', { selectedImage: galleryItem('image', 'a.png') });
    const healed = rebuildProjectMediaUrls(selectedOnly, urls);
    expect(healed).not.toBe(selectedOnly);
    expect(healed.canvas).toBe(selectedOnly.canvas);
    expect(healed.widgetInstances.generate).toBe(selectedOnly.widgetInstances.generate);
  });

  it('leaves an item without a name untouched', () => {
    const nameless = { fullUrl: `${STALE}/images/i/x.png/full`, kind: 'image', name: '' };
    const project = withGallery(createDraftProject([]), 'gallery', { selectedImage: nameless });

    expect(galleryValues(rebuildProjectMediaUrls(project, urls), 'gallery').selectedImage).toBe(nameless);
  });
});
