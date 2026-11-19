import { afterEach, describe, expect, it, vi } from 'vitest';

import { INLINE_EDIT_SELECTOR, shouldFocusCanvasSurface } from './surfaceFocus';

class FakeElement {
  parent: FakeElement | null = null;

  constructor(private readonly matchesInlineEdit = false) {}

  appendTo(parent: FakeElement): this {
    this.parent = parent;
    return this;
  }

  closest(selector: string): FakeElement | null {
    expect(selector).toBe(INLINE_EDIT_SELECTOR);
    if (this.matchesInlineEdit) {
      return this;
    }
    return this.parent ? this.parent.closest(selector) : null;
  }

  contains(other: unknown): boolean {
    let current = other as FakeElement | null;
    while (current) {
      if (current === this) {
        return true;
      }
      current = current.parent;
    }
    return false;
  }
}

const asContainer = (el: FakeElement): HTMLElement => el as unknown as HTMLElement;
const asTarget = (el: FakeElement): EventTarget => el as unknown as EventTarget;
const asActive = (el: FakeElement): Element => el as unknown as Element;

describe('shouldFocusCanvasSurface', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('focuses when DOM focus is outside the surface (e.g. a layers-panel button)', () => {
    vi.stubGlobal('Element', FakeElement);
    const container = new FakeElement();
    const overlayCanvas = new FakeElement().appendTo(container);
    const layersButtonElsewhere = new FakeElement();

    expect(
      shouldFocusCanvasSurface(asContainer(container), asTarget(overlayCanvas), asActive(layersButtonElsewhere))
    ).toBe(true);
  });

  it('focuses when nothing is focused at all', () => {
    vi.stubGlobal('Element', FakeElement);
    const container = new FakeElement();
    const overlayCanvas = new FakeElement().appendTo(container);

    expect(shouldFocusCanvasSurface(asContainer(container), asTarget(overlayCanvas), null)).toBe(true);
  });

  it('does not refocus when focus is already inside the surface (open text session: click-away commits via the engine)', () => {
    vi.stubGlobal('Element', FakeElement);
    const container = new FakeElement();
    const overlayCanvas = new FakeElement().appendTo(container);
    const contenteditable = new FakeElement(true).appendTo(container);

    expect(shouldFocusCanvasSurface(asContainer(container), asTarget(overlayCanvas), asActive(contenteditable))).toBe(
      false
    );
  });

  it('does not steal focus when the click lands inside an inline editor', () => {
    vi.stubGlobal('Element', FakeElement);
    const container = new FakeElement();
    const portalEditor = new FakeElement(true);
    const focusElsewhere = new FakeElement();

    expect(shouldFocusCanvasSurface(asContainer(container), asTarget(portalEditor), asActive(focusElsewhere))).toBe(
      false
    );
  });

  it('focuses when the target is not an Element (defensive: non-element event targets)', () => {
    vi.stubGlobal('Element', FakeElement);
    const container = new FakeElement();

    expect(shouldFocusCanvasSurface(asContainer(container), {} as EventTarget, null)).toBe(true);
  });
});
