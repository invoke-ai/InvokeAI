import { describe, expect, it } from 'vitest';

import { addOutputImageNames, getFirstOutputImageName, getOutputImageNames } from './outputImages';

describe('getOutputImageNames', () => {
  it('reads direct images, collections, and workflow-return values independent of names', () => {
    expect(
      getOutputImageNames({
        collection: [{ image_name: 'a.png' }, { image_name: 'b.png' }],
      })
    ).toEqual(['a.png', 'b.png']);

    expect(
      getOutputImageNames({
        type: 'workflow_return_output',
        values: {
          collection: {
            collection: [{ image_name: 'c.png' }],
            type: 'image_collection_output',
          },
          image: { image: { image_name: 'd.png' }, type: 'image_output' },
          values: { image_name: 'e.png' },
        },
      })
    ).toEqual(['c.png', 'd.png', 'e.png']);

    expect(
      getOutputImageNames({
        type: 'workflow_return_output',
        values: { Images: { collection: [{ image_name: 'legacy.png' }] } },
      })
    ).toEqual(['legacy.png']);
  });

  it('does not inspect metadata, controls, or nested input objects', () => {
    expect(
      getOutputImageNames({
        input: { image: { image_name: 'input.png' } },
        metadata: { image: { image_name: 'metadata.png' } },
        output_meta: { image: { image_name: 'meta-output.png' } },
        values: { result: { image_name: 'result.png' } },
      })
    ).toEqual(['result.png']);
  });

  it('accepts return keys named image, collection, and values', () => {
    expect(
      getOutputImageNames({
        values: {
          collection: [{ image_name: 'collection.png' }],
          image: { image_name: 'image.png' },
          values: { image_name: 'values.png' },
        },
      })
    ).toEqual(['collection.png', 'image.png', 'values.png']);
  });

  it('adds names into a caller-owned set without changing their order', () => {
    const names = new Set(['existing.png']);

    addOutputImageNames(
      {
        type: 'workflow_return_output',
        values: {
          First: { image_name: 'first.png' },
          Duplicate: { image_name: 'first.png' },
          Second: { image_name: 'second.png' },
        },
      },
      names
    );

    expect([...names]).toEqual(['existing.png', 'first.png', 'second.png']);
  });

  it('stops after the first image when only a thumbnail is needed', () => {
    let secondImageVisited = false;
    const collection = [
      { image_name: 'first.png' },
      {
        get image_name() {
          secondImageVisited = true;
          return 'second.png';
        },
      },
    ];

    expect(getFirstOutputImageName({ type: 'image_collection_output', collection })).toBe('first.png');
    expect(secondImageVisited).toBe(false);
  });
});
