import { describe, expect, it } from 'vitest';

import {
  getWorkflowMediaFieldDropId,
  getWorkflowMediaFieldDropItem,
  getWorkflowMediaFieldDropItems,
} from './mediaFieldDnd';

const drag = (items: { kind: string; name: string }[]) => ({ items, kind: 'gallery-item' });

describe('getWorkflowMediaFieldDropItem', () => {
  it('accepts a single gallery item of the matching kind', () => {
    expect(getWorkflowMediaFieldDropItem(drag([{ kind: 'video', name: 'clip.mp4' }]), 'video')).toEqual({
      kind: 'video',
      name: 'clip.mp4',
    });
    expect(getWorkflowMediaFieldDropItem(drag([{ kind: 'image', name: 'a.png' }]), 'image')).toEqual({
      kind: 'image',
      name: 'a.png',
    });
  });

  it('rejects kind mismatches', () => {
    expect(getWorkflowMediaFieldDropItem(drag([{ kind: 'image', name: 'a.png' }]), 'video')).toBeNull();
    expect(getWorkflowMediaFieldDropItem(drag([{ kind: 'video', name: 'clip.mp4' }]), 'image')).toBeNull();
  });

  it('rejects multi-item drags outright rather than keeping the first item', () => {
    expect(
      getWorkflowMediaFieldDropItem(
        drag([
          { kind: 'video', name: 'a.mp4' },
          { kind: 'video', name: 'b.mp4' },
        ]),
        'video'
      )
    ).toBeNull();
  });

  it('rejects payloads that are not gallery item drags', () => {
    expect(getWorkflowMediaFieldDropItem(null, 'video')).toBeNull();
    expect(getWorkflowMediaFieldDropItem({ kind: 'gallery-board' }, 'video')).toBeNull();
    expect(getWorkflowMediaFieldDropItem(drag([]), 'video')).toBeNull();
  });
});

describe('getWorkflowMediaFieldDropId', () => {
  it('namespaces ids so they cannot collide with other droppables', () => {
    expect(getWorkflowMediaFieldDropId('node-1-video:r1')).toBe('workflow-media-field:node-1-video:r1');
  });
});

describe('getWorkflowMediaFieldDropItems', () => {
  it('returns every dragged item of the kind, in drag order', () => {
    expect(
      getWorkflowMediaFieldDropItems(
        drag([
          { kind: 'image', name: 'a.png' },
          { kind: 'image', name: 'b.png' },
        ]),
        'image'
      )
    ).toEqual([
      { kind: 'image', name: 'a.png' },
      { kind: 'image', name: 'b.png' },
    ]);
  });

  it('rejects mixed-kind drags and non-gallery payloads', () => {
    expect(
      getWorkflowMediaFieldDropItems(
        drag([
          { kind: 'image', name: 'a.png' },
          { kind: 'video', name: 'b.mp4' },
        ]),
        'image'
      )
    ).toEqual([]);
    expect(getWorkflowMediaFieldDropItems({ kind: 'gallery-board' }, 'image')).toEqual([]);
  });
});
