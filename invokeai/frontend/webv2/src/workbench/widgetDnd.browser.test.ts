import { afterEach, describe, expect, it } from 'vitest';

import { measureDroppableVisibleRect } from './widgetDnd';

let host: HTMLDivElement | null = null;

const mount = (html: string): HTMLDivElement => {
  host = document.createElement('div');
  host.style.cssText = 'left:0;position:fixed;top:0;';
  host.innerHTML = html;
  document.body.append(host);

  return host;
};

afterEach(() => {
  host?.remove();
  host = null;
});

describe('measureDroppableVisibleRect', () => {
  it('returns the raw client rect when nothing clips the droppable', () => {
    const root = mount('<div id="target" style="height:40px;width:120px;"></div>');
    const target = root.querySelector<HTMLElement>('#target')!;

    expect(measureDroppableVisibleRect(target)).toMatchObject({ height: 40, width: 120 });
  });

  it('collapses a droppable scrolled out of its container to zero height', () => {
    // Mirrors a gallery board row below the fold: its client rect would
    // otherwise extend past the list over whatever is rendered below.
    const root = mount(`
      <div id="scroller" style="height:100px;overflow-y:auto;width:200px;">
        <div style="height:120px;"></div>
        <div id="target" style="height:40px;"></div>
      </div>
    `);
    const target = root.querySelector<HTMLElement>('#target')!;
    const scroller = root.querySelector<HTMLElement>('#scroller')!;
    const rect = measureDroppableVisibleRect(target);
    const clip = scroller.getBoundingClientRect();

    expect(rect.height).toBe(0);
    expect(rect.top).toBe(clip.bottom);
    expect(rect.bottom).toBe(clip.bottom);
    // Width axis is not clipped by an overflow-y container.
    expect(rect.width).toBe(scroller.clientWidth);
  });

  it('keeps only the visible slice of a partially scrolled droppable', () => {
    const root = mount(`
      <div id="scroller" style="height:100px;overflow-y:auto;width:200px;">
        <div style="height:80px;"></div>
        <div id="target" style="height:40px;"></div>
      </div>
    `);
    const target = root.querySelector<HTMLElement>('#target')!;
    const scroller = root.querySelector<HTMLElement>('#scroller')!;
    const rect = measureDroppableVisibleRect(target);
    const clip = scroller.getBoundingClientRect();

    expect(rect.height).toBe(20);
    expect(rect.bottom).toBe(clip.bottom);
  });

  it('clips against every overflow ancestor, not just the nearest', () => {
    // The stacked gallery wraps the board panel in an overflow:hidden box
    // that is itself shorter than the scroll area it contains.
    const root = mount(`
      <div style="height:60px;overflow:hidden;width:200px;">
        <div id="scroller" style="height:100px;overflow-y:auto;">
          <div style="height:70px;"></div>
          <div id="target" style="height:40px;"></div>
        </div>
      </div>
    `);
    const target = root.querySelector<HTMLElement>('#target')!;
    const rect = measureDroppableVisibleRect(target);

    expect(rect.height).toBe(0);
  });

  it('tracks the container scrolling the droppable into view', () => {
    const root = mount(`
      <div id="scroller" style="height:100px;overflow-y:auto;width:200px;">
        <div style="height:120px;"></div>
        <div id="target" style="height:40px;"></div>
      </div>
    `);
    const target = root.querySelector<HTMLElement>('#target')!;
    const scroller = root.querySelector<HTMLElement>('#scroller')!;

    scroller.scrollTop = 60;

    expect(measureDroppableVisibleRect(target).height).toBe(40);
  });
});
