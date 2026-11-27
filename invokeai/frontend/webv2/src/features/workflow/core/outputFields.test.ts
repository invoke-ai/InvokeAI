import { describe, expect, it } from 'vitest';

import { formatOutputFieldValue, getOutputFieldNamesByScope, getOutputFieldRows } from './outputFields';

describe('output field scopes', () => {
  it('groups visible outputs and keeps hidden scheduler outputs out of the node body', () => {
    const fields = [
      {
        name: 'loop_linkage',
        title: 'Linkage',
        description: '',
        type: { name: 'AnyField', cardinality: 'SINGLE' as const, batch: false },
      },
      {
        name: 'item',
        title: 'Item',
        description: '',
        outputScope: 'iteration' as const,
        type: { name: 'AnyField', cardinality: 'SINGLE' as const, batch: false },
      },
      {
        name: 'output',
        title: 'Output',
        description: '',
        uiHidden: true,
        type: { name: 'AnyField', cardinality: 'SINGLE' as const, batch: false },
      },
      {
        name: 'output_collection',
        title: 'Collection',
        description: '',
        outputScope: 'final' as const,
        type: { name: 'CollectionField', cardinality: 'COLLECTION' as const, batch: false },
      },
    ];

    const names = getOutputFieldNamesByScope(fields);

    expect(names).toEqual({
      all: ['loop_linkage', 'item', 'output_collection'],
      final: ['output_collection'],
      iteration: ['item'],
      unscoped: ['loop_linkage'],
    });
    expect(getOutputFieldRows(names)).toEqual([
      { fieldName: 'loop_linkage', type: 'field' },
      { scope: 'iteration', type: 'header' },
      { fieldName: 'item', type: 'field' },
      { scope: 'final', type: 'header' },
      { fieldName: 'output_collection', type: 'field' },
    ]);
  });
});

describe('output value formatting', () => {
  it('formats primitives, named objects, collections, and truncates long strings', () => {
    expect(formatOutputFieldValue({ type: 'integer_output', value: 7 }, 'value')).toEqual({ full: '7', short: '7' });
    expect(formatOutputFieldValue({ value: 0.123456 }, 'value')?.short).toBe('0.1235');
    expect(formatOutputFieldValue({ image: { image_name: 'a.png', width: 1 } }, 'image')?.short).toBe('a.png');
    expect(formatOutputFieldValue({ latents: { latents_name: 'lat-1' } }, 'latents')?.short).toBe('lat-1');
    expect(formatOutputFieldValue({ collection: [1, 2, 3] }, 'collection')?.short).toBe('3 items');
    expect(formatOutputFieldValue({ color: { a: 255, b: 3, g: 2, r: 1 } }, 'color')?.short).toBe('rgba(1, 2, 3, 255)');
    expect(formatOutputFieldValue({ value: null }, 'value')?.short).toBe('—');

    const long = formatOutputFieldValue({ value: 'x'.repeat(60) }, 'value');

    expect(long?.short).toHaveLength(40);
    expect(long?.full).toHaveLength(60);
  });

  it('returns null when the result has no such field', () => {
    expect(formatOutputFieldValue({ type: 'integer_output' }, 'value')).toBeNull();
    expect(formatOutputFieldValue('not-an-object', 'value')).toBeNull();
  });
});
