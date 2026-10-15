import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ProjectsToolbar } from './ProjectsToolbar';

/**
 * Verify both selected styling and keyboard tooltips: asChild on the item overwrites checked data-state, while an
 * icon-only trigger cannot receive focus.
 */

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const render = async (overrides: { view?: 'grid' | 'list'; onViewChange?: (view: 'grid' | 'list') => void } = {}) => {
  host = document.createElement('div');
  host.style.width = '720px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <ProjectsToolbar
          searchTerm=""
          sort="edited"
          view={overrides.view ?? 'grid'}
          onSearchTermChange={vi.fn()}
          onSortChange={vi.fn()}
          onViewChange={overrides.onViewChange ?? vi.fn()}
        />
      </ChakraProvider>
    );
  });

  return host;
};

const settle = async () => {
  await act(async () => {
    await new Promise((resolve) => {
      setTimeout(resolve, 60);
    });
  });
};

/** The tooltip fades in and out, so its visibility is polled rather than sampled. */
const waitUntil = async (predicate: () => boolean, description: string) => {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (predicate()) {
      return;
    }

    await settle();
  }

  throw new Error(`never became true: ${description}`);
};

/** Select by PROJECTS_VIEW_IDS order because tests render untranslated keys. */
const viewRadios = (container: HTMLElement): HTMLInputElement[] => {
  const radios = [...container.querySelectorAll<HTMLInputElement>('input[type="radio"]')];

  expect(radios).toHaveLength(2);

  return radios;
};

const tooltipContent = (): HTMLElement | null =>
  document.querySelector<HTMLElement>('[data-scope="tooltip"][data-part="content"]');

/**
 * Asked as "can someone see it", not as a `data-state` reading: the node is
 * lazily mounted and then kept, and its state attribute is not what decides
 * whether the label is on screen.
 */
const isTooltipVisible = (): boolean => tooltipContent()?.checkVisibility() ?? false;

describe('ProjectsToolbar layout control', () => {
  it('puts the accessible name on the control that actually takes focus', async () => {
    const [grid, list] = viewRadios(await render());

    expect(grid!.getAttribute('aria-label')).toBeTruthy();
    expect(list!.getAttribute('aria-label')).toBeTruthy();
    expect(grid!.getAttribute('aria-label')).not.toBe(list!.getAttribute('aria-label'));
  });

  it('keeps the selected segment marked as checked', async () => {
    const [grid, list] = viewRadios(await render({ view: 'list' }));

    expect(list!.closest('label')?.dataset.state).toBe('checked');
    expect(grid!.closest('label')?.dataset.state).toBe('unchecked');
  });

  /**
   * The indicator is positioned from the checked item's measured box, so a
   * `data-state` clobbered by a tooltip trigger collapses it to nothing.
   */
  it('gives the selection indicator a measurable box', async () => {
    const container = await render({ view: 'grid' });

    await settle();

    const indicator = container.querySelector<HTMLElement>('[data-scope="segment-group"][data-part="indicator"]');

    expect(indicator).not.toBeNull();
    expect(indicator!.getBoundingClientRect().width).toBeGreaterThan(0);
  });

  it('shows the label on keyboard focus, not only on pointer hover', async () => {
    const container = await render();
    const [, list] = viewRadios(container);

    expect(isTooltipVisible()).toBe(false);

    await act(() => {
      list!.focus();
    });
    await waitUntil(isTooltipVisible, 'the tooltip appeared on focus');

    expect(tooltipContent()?.textContent).toContain(list!.getAttribute('aria-label'));
  });

  it('hides the label again when focus leaves', async () => {
    const container = await render();
    const [, list] = viewRadios(container);

    await act(() => {
      list!.focus();
    });
    await waitUntil(isTooltipVisible, 'the tooltip appeared on focus');

    await act(() => {
      list!.blur();
    });
    await waitUntil(() => !isTooltipVisible(), 'the tooltip went away on blur');
  });
});
