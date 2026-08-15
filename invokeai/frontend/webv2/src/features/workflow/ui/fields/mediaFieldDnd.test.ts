import { describe, expect, it } from 'vitest';

import { getWorkflowMediaFieldDropId, getWorkflowMediaFieldDropItem } from './mediaFieldDnd';

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
