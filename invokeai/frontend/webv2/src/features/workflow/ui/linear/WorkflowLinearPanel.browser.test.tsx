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

  const renderToggle = async () => {
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

    return [...host.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
  };

  const selection = (tabs: HTMLButtonElement[]) => tabs.map((tab) => tab.getAttribute('aria-selected'));

  it('exposes View and Edit as a labelled tablist', async () => {
    const tabs = await renderToggle();

    expect(tabs).toHaveLength(2);
    expect(host.querySelector('[role="tablist"]')?.getAttribute('aria-label')).toBeTruthy();
    expect(selection(tabs)).toEqual(['true', 'false']);
    expect(
      tabs
        .map((tab) => tab.getAttribute('aria-controls'))
        .filter((id): id is string => id !== null)
        .map((id) => document.getElementById(id))
    ).not.toContain(null);
  });

  it('activates View and Edit with pointer and arrow keys', async () => {
    const tabs = await renderToggle();

    await act(() => userEvent.click(tabs[1]!));
    expect(selection(tabs)).toEqual(['false', 'true']);

    // Roving focus: the tablist is one tab stop and arrows move within it.
    tabs[1]?.focus();
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    expect(selection(tabs)).toEqual(['true', 'false']);

    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(selection(tabs)).toEqual(['false', 'true']);
  });
});
