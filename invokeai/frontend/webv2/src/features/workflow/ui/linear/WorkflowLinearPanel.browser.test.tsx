import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { PanelModeToggle } from './WorkflowLinearPanel';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('Workflow Linear panel mode toggle', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('activates View and Edit with pointer and keyboard button semantics', async () => {
    const Harness = () => {
      const [mode, setMode] = useState<'view' | 'edit'>('view');
      return <PanelModeToggle mode={mode} onChange={setMode} />;
    };

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    const buttons = host.querySelectorAll<HTMLButtonElement>('button[aria-pressed]');
    expect([...buttons].map((button) => button.getAttribute('aria-pressed'))).toEqual(['true', 'false']);

    await act(() => userEvent.click(buttons[1]!));
    expect([...buttons].map((button) => button.getAttribute('aria-pressed'))).toEqual(['false', 'true']);

    buttons[0]?.focus();
    await act(() => userEvent.keyboard('{Enter}'));
    expect([...buttons].map((button) => button.getAttribute('aria-pressed'))).toEqual(['true', 'false']);

    await act(() => userEvent.tab());
    expect(document.activeElement).toBe(buttons[1]);

    await act(() => userEvent.keyboard(' '));
    expect([...buttons].map((button) => button.getAttribute('aria-pressed'))).toEqual(['false', 'true']);
  });
});
