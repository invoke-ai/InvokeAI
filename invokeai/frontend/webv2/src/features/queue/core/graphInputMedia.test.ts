import { describe, expect, it } from 'vitest';

import { collectGraphInputMediaNames } from './graphInputMedia';

describe('collectGraphInputMediaNames', () => {
  it('collects image and video names from node field values, including nested ones', () => {
    const graph = {
      edges: [],
      id: 'g',
      nodes: {
        first_frame: { id: 'first_frame', image: { image_name: 'keyframe.png' }, type: 'image' },
        ref_collection: {
          id: 'ref_collection',
          type: 'image_collection',
          value: [{ image_name: 'ref-1.png' }, { image_name: 'ref-2.png' }],
        },
        source_clip: { id: 'source_clip', type: 'video', video: { video_name: 'input.mp4' } },
      },
    };

    const { imageNames, videoNames } = collectGraphInputMediaNames(graph);

    expect([...imageNames].sort()).toEqual(['keyframe.png', 'ref-1.png', 'ref-2.png']);
    expect([...videoNames]).toEqual(['input.mp4']);
  });

  it('returns empty sets for malformed or legacy graphs', () => {
    for (const graph of [undefined, null, 'not-a-graph', 42, {}, { nodes: null }, { nodes: 'oops' }]) {
      const { imageNames, videoNames } = collectGraphInputMediaNames(graph);
      expect(imageNames.size).toBe(0);
      expect(videoNames.size).toBe(0);
    }
  });

  it('ignores non-string name values', () => {
    const graph = {
      nodes: {
        weird: { id: 'weird', image: { image_name: 7 }, type: 'image', video: { video_name: null } },
      },
    };

    const { imageNames, videoNames } = collectGraphInputMediaNames(graph);

    expect(imageNames.size).toBe(0);
    expect(videoNames.size).toBe(0);
  });
});
