/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { GenerateCollapsibleSection } from './GenerateCollapsibleSection';
import { GenerationSectionJumpMenu } from './GenerationSectionJumpMenu';
import { GenerationSectionPanel } from './sectionPanel';

const preferences = vi.hoisted(() => {
  let sectionsOpen: Readonly<Record<string, boolean>> = {};
  const listeners = new Set<() => void>();

  return {
    get: () => sectionsOpen,
    reset: (next: Readonly<Record<string, boolean>>) => {
      sectionsOpen = next;
    },
    setSectionsOpen: (changes: Readonly<Record<string, boolean>>) => {
      sectionsOpen = { ...sectionsOpen, ...changes };
      listeners.forEach((listener) => listener());
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
});
const copied = vi.hoisted(() => ({ text: [] as string[] }));

vi.mock('@features/generation/ui/GenerationUiContext', async () => {
  const { useSyncExternalStore } = await import('react');

  return {
    useGenerationUi: () => ({
      sectionPreferences: {
        sectionsOpen: useSyncExternalStore(preferences.subscribe, preferences.get),
        setSectionsOpen: preferences.setSectionsOpen,
      },
    }),
  };
});

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => (values ? `${key} ${JSON.stringify(values)}` : key),
  }),
}));

vi.mock('@platform/browser/clipboard', () => ({
  canCopyText: () => true,
  copyText: (text: string) => {
    copied.text.push(text);
    return Promise.resolve();
  },
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

beforeEach(() => {
  preferences.reset({});
  copied.text = [];
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const render = async (children: ReactNode) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(<ChakraProvider value={system}>{children}</ChakraProvider>);
  });
};

const section = (label: string) =>
  [...document.querySelectorAll<HTMLElement>('.generate-section')].find(
    (candidate) => candidate.querySelector('[data-section-chip]')?.textContent?.startsWith(label) ?? false
  )!;
const title = (label: string) => section(label).querySelector<HTMLButtonElement>(':scope > [data-part="trigger"]')!;
const states = (...labels: string[]) => labels.map((label) => section(label).dataset.state);

const renderTwoPanels = () =>
  render(
    <>
      <GenerationSectionPanel id="generate">
        <GenerateCollapsibleSection defaultOpen label="Size" sectionId="size">
          <input aria-label="Width" />
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection label="Render" sectionId="render">
          <div>steps</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection defaultOpen label="Fill">
          <div>fill</div>
        </GenerateCollapsibleSection>
      </GenerationSectionPanel>
      <GenerationSectionPanel id="upscale">
        <GenerateCollapsibleSection defaultOpen label="Source" sectionId="source">
          <div>source</div>
        </GenerateCollapsibleSection>
      </GenerationSectionPanel>
    </>
  );

describe('GenerateCollapsibleSection', () => {
  it('solos a section on Alt-click, within its own panel only, persisting the persisted ones in one write', async () => {
    const write = vi.spyOn(preferences, 'setSectionsOpen');
    await renderTwoPanels();

    await act(() => userEvent.click(title('Render'), { modifiers: ['Alt'] }));

    expect(states('Size', 'Render', 'Fill', 'Source')).toEqual(['closed', 'open', 'closed', 'open']);
    expect(write).toHaveBeenCalledOnce();
    expect(write).toHaveBeenCalledWith({ render: true, size: false });
    write.mockRestore();
  });

  it('moves between titles with the arrow, Home and End keys, and opens and folds with right and left', async () => {
    await renderTwoPanels();
    title('Size').focus();

    await userEvent.keyboard('{ArrowDown}');
    expect(document.activeElement).toBe(title('Render'));
    await userEvent.keyboard('{End}');
    expect(document.activeElement).toBe(title('Fill'));
    await userEvent.keyboard('{ArrowDown}');
    expect(document.activeElement).toBe(title('Fill'));
    await userEvent.keyboard('{Home}');
    expect(document.activeElement).toBe(title('Size'));
    await userEvent.keyboard('{ArrowUp}');
    expect(document.activeElement).toBe(title('Size'));

    await userEvent.keyboard('{ArrowLeft}');
    expect(states('Size')).toEqual(['closed']);
    await userEvent.keyboard('{ArrowRight}');
    expect(states('Size')).toEqual(['open']);
    await userEvent.keyboard('{Enter}');
    expect(states('Size')).toEqual(['closed']);
  });

  it('leaves arrow keys typed in a field to the field', async () => {
    await renderTwoPanels();
    const width = page.getByLabelText('Width');

    await userEvent.click(width);
    await userEvent.keyboard('{ArrowDown}{ArrowLeft}');

    expect(document.activeElement).toBe(width.element());
    expect(states('Size')).toEqual(['open']);
  });

  it('offers the panel actions, the summary copy and the reset from the title context menu', async () => {
    const reset = vi.fn();
    await render(
      <GenerationSectionPanel>
        <GenerateCollapsibleSection defaultOpen label="Size" sectionId="size">
          <div>body</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection
          defaultOpen
          isChanged
          label="Render"
          sectionId="render"
          summary={['30', 'CFG 7']}
          onReset={reset}
        >
          <div>body</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection isChanged={false} label="Advanced" sectionId="advanced" onReset={reset}>
          <div>body</div>
        </GenerateCollapsibleSection>
      </GenerationSectionPanel>
    );
    const openMenu = async (label: string) => {
      await act(() => {
        title(label).dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
      });
      await expect.element(page.getByRole('menu')).toBeVisible();
      // The menu takes focus once it has registered its dismiss handlers.
      await expect.poll(() => document.activeElement?.getAttribute('role')).toBe('menu');
    };
    const item = (name: string) => page.getByRole('menuitem', { name });

    await openMenu('Render');
    await expect.element(item('widgets.generate.sectionMenu.expandAll')).toBeVisible();
    await act(() => userEvent.click(item('widgets.generate.sectionMenu.copySummary')));
    expect(copied.text).toEqual(['widgets.generate.sectionMenu.summaryText {"label":"Render","summary":"30 · CFG 7"}']);

    await openMenu('Render');
    await act(() => userEvent.click(item('widgets.generate.sectionMenu.reset')));
    expect(reset).toHaveBeenCalledOnce();

    title('Render').focus();
    await openMenu('Render');
    await userEvent.keyboard('{Home}{Enter}');
    expect(states('Size', 'Render', 'Advanced')).toEqual(['closed', 'open', 'closed']);
    await expect.poll(() => document.activeElement).toBe(title('Render'));

    await openMenu('Advanced');
    await expect.element(item('widgets.generate.sectionMenu.copySummary')).not.toBeInTheDocument();
    await expect.element(item('widgets.generate.sectionMenu.reset')).toHaveAttribute('data-disabled');
    await act(() => userEvent.click(item('widgets.generate.sectionMenu.expandAll')));
    expect(states('Size', 'Render', 'Advanced')).toEqual(['open', 'open', 'open']);

    await openMenu('Size');
    await expect.element(item('widgets.generate.sectionMenu.foldAll')).toBeVisible();
    await expect.element(item('widgets.generate.sectionMenu.reset')).not.toBeInTheDocument();
  });
});
describe('GenerationSectionJumpMenu', () => {
  const jumpButton = () => page.getByRole('button', { name: 'widgets.generate.jumpToSection' });

  it('lists the panel sections in order and opens, scrolls to and focuses the one chosen', async () => {
    await render(
      <>
        <GenerationSectionJumpMenu panelId="generate" />
        <div style={{ height: 160, overflowY: 'auto' }} data-testid="scroller">
          <GenerationSectionPanel id="generate">
            <GenerateCollapsibleSection defaultOpen label="Size" sectionId="size">
              <div style={{ height: 400 }}>size</div>
            </GenerateCollapsibleSection>
            <GenerateCollapsibleSection label="Render" sectionId="render">
              <div style={{ height: 400 }}>render</div>
            </GenerateCollapsibleSection>
          </GenerationSectionPanel>
        </div>
        <GenerationSectionPanel id="upscale">
          <GenerateCollapsibleSection label="Source" sectionId="source">
            <div>source</div>
          </GenerateCollapsibleSection>
        </GenerationSectionPanel>
      </>
    );

    await act(() => userEvent.click(jumpButton()));
    const items = page.getByRole('menuitem');
    await expect.element(items.first()).toBeVisible();
    expect(items.elements().map((item) => item.textContent)).toEqual(['Size', 'Render']);

    await act(() => userEvent.click(page.getByRole('menuitem', { name: 'Render' })));

    expect(section('Render').dataset.state).toBe('open');
    await expect.poll(() => document.activeElement).toBe(title('Render'));
    const scroller = page.getByTestId('scroller').element();
    await expect
      .poll(() => Math.round(section('Render').getBoundingClientRect().top - scroller.getBoundingClientRect().top))
      .toBe(0);
  });
});
