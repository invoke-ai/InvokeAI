import { afterEach, describe, expect, it, vi } from 'vitest';

import { getHotkeyTargetWidget } from './targetWidget';

class FakeElement {
  constructor(private readonly attrs: Record<string, string> | null) {}

  closest(): FakeElement | null {
    return this.attrs ? this : null;
  }

  getAttribute(name: string): string | null {
    return this.attrs?.[name] ?? null;
  }
}

describe('getHotkeyTargetWidget', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads the nearest hotkey widget shell from the event target', () => {
    vi.stubGlobal('Element', FakeElement);

    expect(
      getHotkeyTargetWidget(
        new FakeElement({
          'data-hotkey-widget-instance-id': 'preview',
          'data-hotkey-widget-type-id': 'preview',
        }) as unknown as EventTarget
      )
    ).toEqual({ instanceId: 'preview', typeId: 'preview' });
  });

  it('ignores targets outside widget shells', () => {
    vi.stubGlobal('Element', FakeElement);

    expect(getHotkeyTargetWidget(new FakeElement(null) as unknown as EventTarget)).toBeNull();
  });
});
