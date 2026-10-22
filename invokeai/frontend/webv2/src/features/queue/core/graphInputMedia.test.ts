import { describe, expect, it } from 'vitest';

import { collectSubmissionInputMediaNames } from './graphInputMedia';

describe('collectSubmissionInputMediaNames', () => {
  it('collects image and video names from node field values, including nested ones', () => {
    const submission = {
      graph: {
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
      },
    };

    const { imageNames, videoNames } = collectSubmissionInputMediaNames(submission);

    expect([...imageNames].sort()).toEqual(['keyframe.png', 'ref-1.png', 'ref-2.png']);
    expect([...videoNames]).toEqual(['input.mp4']);
  });

  it('collects image names from every group and datum of workflow batch data', () => {
    const submission = {
      batchData: [
        [
          { fieldName: 'image', items: [{ image_name: 'a.png' }, { image_name: 'b.png' }], nodeId: 'primitive_a' },
          { fieldName: 'value', items: ['prompt one', 'prompt two'], nodeId: 'prompt' },
        ],
        [{ fieldName: 'image', items: [{ image_name: 'c.png' }], nodeId: 'primitive_b' }],
      ],
      graph: { edges: [], id: 'g', nodes: { primitive_a: { id: 'primitive_a', type: 'image' } } },
      kind: 'workflow',
    };

    const { imageNames, videoNames } = collectSubmissionInputMediaNames(submission);

    expect([...imageNames].sort()).toEqual(['a.png', 'b.png', 'c.png']);
    expect(videoNames.size).toBe(0);
  });

  it('returns empty sets for malformed or legacy submissions', () => {
    for (const submission of [
      undefined,
      null,
      'not-a-submission',
      42,
      {},
      { graph: null },
      { graph: { nodes: null } },
      { graph: { nodes: 'oops' } },
      { batchData: null, graph: { nodes: {} } },
      { batchData: 'oops', graph: { nodes: {} } },
      { batchData: [null, 7, [null, 'datum']], graph: { nodes: {} } },
    ]) {
      const { imageNames, videoNames } = collectSubmissionInputMediaNames(submission);
      expect(imageNames.size).toBe(0);
      expect(videoNames.size).toBe(0);
    }
  });

  it('ignores non-string name values', () => {
    const submission = {
      batchData: [[{ fieldName: 'image', items: [{ image_name: 3 }], nodeId: 'weird' }]],
      graph: {
        nodes: {
          weird: { id: 'weird', image: { image_name: 7 }, type: 'image', video: { video_name: null } },
        },
      },
    };

    const { imageNames, videoNames } = collectSubmissionInputMediaNames(submission);

    expect(imageNames.size).toBe(0);
    expect(videoNames.size).toBe(0);
  });
});
