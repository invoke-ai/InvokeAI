import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { useExtensionHotkeyDefinitions } from './extensionHotkeys';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Harness = () => {
  const hotkeys = useExtensionHotkeyDefinitions();

  return <output>{hotkeys.length}</output>;
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('useExtensionHotkeyDefinitions', () => {
  it('returns no extension hotkeys outside a WorkbenchProvider', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => root?.render(<Harness />));

    expect(host.querySelector('output')?.textContent).toBe('0');
  });
});
