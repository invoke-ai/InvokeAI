/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { GenerationIssue } from '@features/generation/core/generationIssues';

import { ChakraProvider } from '@chakra-ui/react';
import { revealGenerateIssue } from '@features/generation/ui/revealGenerateIssue';
import { system } from '@theme/system';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { GenerateCollapsibleSection } from './GenerateCollapsibleSection';
import { GenerateIssueNotes, GenerateIssuesProvider } from './GenerateIssues';
import { GenerationSectionJumpMenu } from './GenerationSectionJumpMenu';
import { GenerationSectionPanel } from './sectionPanel';

/** The user's persisted open states, as the preference port reads and writes them. */
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

// Keys stand in for copy; values show what each string was built from.
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

/** Two panels side by side, as Generate and Upscale can be: persisted and uncontrolled sections in the first. */
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
  it('styles by DOM state so uncontrolled sections (no sectionId) still tell open from closed', async () => {
    // defaultOpen-only consumers require DOM-state styling.
    await render(
      <>
        <GenerateCollapsibleSection defaultOpen label="Open">
          <div>body</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection label="Closed">
          <div>body</div>
        </GenerateCollapsibleSection>
      </>
    );

    expect(title('Open').dataset.state).toBe('open');
    expect(title('Closed').dataset.state).toBe('closed');
    expect(getComputedStyle(title('Open')).color).not.toBe(getComputedStyle(title('Closed')).color);
  });

  it('solos a section on Alt-click, within its own panel only, persisting the persisted ones in one write', async () => {
    // Spied before rendering, as sections read the port when they render.
    const write = vi.spyOn(preferences, 'setSectionsOpen');
    await renderTwoPanels();

    await act(() => userEvent.click(title('Render'), { modifiers: ['Alt'] }));

    expect(states('Size', 'Render', 'Fill', 'Source')).toEqual(['closed', 'open', 'closed', 'open']);
    expect(write).toHaveBeenCalledOnce();
    expect(write).toHaveBeenCalledWith({ render: true, size: false });
    write.mockRestore();
  });

  it('opens every section on Shift-click while any is folded, then folds them all', async () => {
    await renderTwoPanels();

    await act(() => userEvent.click(title('Fill'), { modifiers: ['Shift'] }));
    expect(states('Size', 'Render', 'Fill', 'Source')).toEqual(['open', 'open', 'open', 'open']);
    expect(preferences.get()).toEqual({ render: true });

    await act(() => userEvent.click(title('Fill'), { modifiers: ['Shift'] }));
    expect(states('Size', 'Render', 'Fill', 'Source')).toEqual(['closed', 'closed', 'closed', 'open']);
    expect(preferences.get()).toEqual({ render: false, size: false });
  });

  it('moves between titles with the arrow, Home and End keys, and opens and folds with right and left', async () => {
    await renderTwoPanels();
    title('Size').focus();

    await userEvent.keyboard('{ArrowDown}');
    expect(document.activeElement).toBe(title('Render'));
    await userEvent.keyboard('{End}');
    expect(document.activeElement).toBe(title('Fill'));
    // The last title in its panel: the next panel's titles are not its rows.
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

  it('shows value summaries only while folded and keeps status badges either way', async () => {
    await render(
      <GenerationSectionPanel>
        <GenerateCollapsibleSection
          defaultOpen
          label="Size"
          sectionId="size"
          status={<span data-testid="status">Shared</span>}
          summary={['1024×1024', null, 'Locked']}
        >
          <div>body</div>
        </GenerateCollapsibleSection>
      </GenerationSectionPanel>
    );
    const summary = () => title('Size').querySelector<HTMLElement>('[data-section-summary]')!;

    expect(summary().getAttribute('aria-hidden')).toBe('true');
    await expect.poll(() => getComputedStyle(summary()).visibility).toBe('hidden');
    expect(title('Size').querySelector('[data-testid="status"]')).not.toBeNull();

    await act(() => userEvent.click(title('Size')));

    expect(summary().getAttribute('aria-hidden')).toBeNull();
    await expect.poll(() => getComputedStyle(summary()).visibility).toBe('visible');
    // Skipped items leave no empty separator behind.
    expect([...summary().children].map((item) => item.textContent)).toEqual(['1024×1024', 'Locked']);
    await expect.element(page.getByRole('button', { name: /Size.*1024×1024.*Locked.*Shared/ })).toBeVisible();
  });

  it('tints the chip and counts the issues inside it, open or folded', async () => {
    const issue = { id: 'size-grid', message: 'Off the grid', target: { section: 'dimensions' } } as GenerationIssue;
    await render(
      <GenerateIssuesProvider applyFix={() => undefined} issues={[issue, { ...issue, id: 'size-other' }]}>
        <GenerateCollapsibleSection label="Size" sectionId="dimensions">
          <div>body</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection label="Render" sectionId="render">
          <div>body</div>
        </GenerateCollapsibleSection>
      </GenerateIssuesProvider>
    );
    const chip = (label: string) => title(label).querySelector<HTMLElement>('[data-section-chip]')!;

    expect(chip('Size').querySelector('[data-section-issues]')?.textContent).toContain('widgets.generate.issueCount');
    expect(getComputedStyle(chip('Size')).backgroundColor).not.toBe(getComputedStyle(chip('Render')).backgroundColor);
    expect(chip('Render').querySelector('[data-section-issues]')).toBeNull();
  });

  it('marks a changed section with a dot that names itself', async () => {
    await render(
      <>
        <GenerateCollapsibleSection isChanged label="Render">
          <div>body</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection isChanged={false} label="Size">
          <div>body</div>
        </GenerateCollapsibleSection>
      </>
    );

    expect(title('Render').querySelector('[data-section-changed]')?.textContent).toBe(
      'widgets.generate.sectionChanged'
    );
    expect(title('Size').querySelector('[data-section-changed]')).toBeNull();
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

    // From the keyboard, as the menu key opens it: focus goes back to the title the menu came from.
    title('Render').focus();
    await openMenu('Render');
    await userEvent.keyboard('{Home}{Enter}');
    expect(states('Size', 'Render', 'Advanced')).toEqual(['closed', 'open', 'closed']);
    await expect.poll(() => document.activeElement).toBe(title('Render'));

    // No summary, no copy; a section already at its defaults offers its reset disabled.
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

/** A short scroller holding one long open section, as a panel scrolled deep into its fields. */
const renderPinnedSection = (fields: ReactNode) =>
  render(
    <div data-hotkey-widget-type-id="generate">
      <div data-testid="scroller" style={{ height: 200, overflowY: 'auto' }}>
        <GenerationSectionPanel>
          <GenerateCollapsibleSection defaultOpen label="Render" sectionId="render">
            {fields}
          </GenerateCollapsibleSection>
        </GenerationSectionPanel>
      </div>
    </div>
  );
const scroller = () => page.getByTestId('scroller').element() as HTMLElement;
const nextFrame = () =>
  new Promise<void>((resolve) => {
    requestAnimationFrame(() => resolve());
  });

describe('GenerateCollapsibleSection pinned title', () => {
  it('stays above positioned fields scrolling beneath it', async () => {
    await renderPinnedSection(
      <>
        <div style={{ height: 40 }} />
        {/* Stands in for a segment thumb or a textarea handle: positioned, z-index 2. */}
        <div data-testid="raised" style={{ height: 80, position: 'relative', zIndex: 2 }} />
        <div style={{ height: 600 }} />
      </>
    );
    // The raised field now spans the whole pinned title.
    await act(() => {
      scroller().scrollTop = 100;
    });
    const rect = title('Render').getBoundingClientRect();
    const raised = page.getByTestId('raised').element().getBoundingClientRect();

    expect(raised.top).toBeLessThan(rect.top);
    expect(raised.bottom).toBeGreaterThan(rect.bottom);

    expect(Math.round(rect.top)).toBe(Math.round(scroller().getBoundingClientRect().top));
    expect(
      title('Render').contains(document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2))
    ).toBe(true);
  });

  it('draws its edge only while pinned', async () => {
    await renderPinnedSection(<div style={{ height: 800 }} />);
    const edge = () => getComputedStyle(title('Render'), '::after').opacity;

    await expect.poll(edge).toBe('0');
    await act(() => {
      scroller().scrollTop = 300;
    });
    await expect.poll(edge).toBe('1');
  });

  it('reveals an issue above the view below the pinned title, not under it', async () => {
    const issue = {
      fix: { kind: 'clampGuidance', value: 7 },
      id: 'render-cfg',
      message: 'Guidance out of range',
      target: { section: 'render' },
    } as GenerationIssue;
    await render(
      <div data-hotkey-widget-type-id="generate">
        <div data-testid="scroller" style={{ height: 200, overflowY: 'auto' }}>
          <GenerateIssuesProvider applyFix={() => undefined} issues={[issue]}>
            <GenerationSectionPanel>
              <GenerateCollapsibleSection defaultOpen label="Render" sectionId="render">
                <GenerateIssueNotes section="render" />
                <div style={{ height: 800 }} />
              </GenerateCollapsibleSection>
            </GenerationSectionPanel>
          </GenerateIssuesProvider>
        </div>
      </div>
    );
    await act(() => {
      scroller().scrollTop = 600;
    });

    revealGenerateIssue(issue);
    const fix = () => document.querySelector<HTMLElement>('[data-issue-fix]')!;
    await expect.poll(() => document.activeElement).toBe(fix());
    await nextFrame();

    const fixRect = fix().getBoundingClientRect();
    expect(fixRect.top).toBeGreaterThanOrEqual(title('Render').getBoundingClientRect().bottom);
    expect(document.elementFromPoint(fixRect.left + 4, fixRect.top + fixRect.height / 2)).toBe(fix());
  });
});

describe('GenerateCollapsibleSection on its own', () => {
  it('is a panel of one: Shift-click toggles it and Fold others has nothing to fold', async () => {
    await render(
      <GenerateCollapsibleSection label="Render" summary={['30']} onReset={() => undefined} resetLabel="Reset steps">
        <div>body</div>
      </GenerateCollapsibleSection>
    );

    await act(() => userEvent.click(title('Render'), { modifiers: ['Shift'] }));
    expect(states('Render')).toEqual(['open']);

    await act(() => {
      title('Render').dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect
      .element(page.getByRole('menuitem', { name: 'widgets.generate.sectionMenu.foldOthers' }))
      .toHaveAttribute('data-disabled');
    // A consumer whose reset covers part of the section names that part.
    await expect.element(page.getByRole('menuitem', { name: 'Reset steps' })).toBeVisible();
  });
});

describe('GenerateCollapsibleSection open motion', () => {
  const content = (label: string) => section(label).querySelector<HTMLElement>(':scope > [data-part="content"]')!;
  const fields = (label: string) => content(label).firstElementChild as HTMLElement;
  const isScale = (animation: Animation) => (animation as CSSAnimation).animationName.startsWith('wb-section-scale');

  it('scales the fields, not the clipped content, in by the same few pixels for a short and a tall section', async () => {
    await render(
      <GenerationSectionPanel>
        <GenerateCollapsibleSection label="Short" sectionId="short">
          <div style={{ height: 80 }} />
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection label="Tall" sectionId="tall">
          <div style={{ height: 700 }} />
        </GenerateCollapsibleSection>
      </GenerationSectionPanel>
    );

    for (const label of ['Short', 'Tall']) {
      await act(() => userEvent.click(title(label)));
      // The collapsible measures the content a frame after opening; the scale reads that size.
      await expect.poll(() => content(label).style.getPropertyValue('--height')).not.toBe('0px');
      // Chromium leaves fields unpainted when the element clipping the animating height also runs a transform
      // animation, so the content runs none.
      expect(content(label).getAnimations().some(isScale)).toBe(false);
      const scale = fields(label).getAnimations().find(isScale)!;

      scale.pause();
      scale.currentTime = 0;
      const rect = fields(label).getBoundingClientRect();

      expect(fields(label).offsetWidth - rect.width).toBeCloseTo(8, 1);
      expect(fields(label).offsetHeight - rect.height).toBeCloseTo(4, 1);
      scale.finish();
    }
  });

  it('starts at full size, never an infinite scale, while the size is still unmeasured', async () => {
    const probe = document.createElement('div');
    probe.style.cssText = '--width: 0px; --height: 0px; animation: wb-section-scale-in 1s paused;';
    await render(<div />);
    host!.append(probe);

    expect(getComputedStyle(probe).transform).toBe('matrix(1, 0, 0, 1, 0, 0)');
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

  it('stays hidden while its panel shows fewer than two sections', async () => {
    await render(
      <>
        <GenerationSectionJumpMenu panelId="upscale" />
        <GenerationSectionPanel id="upscale">
          <GenerateCollapsibleSection label="Source" sectionId="source">
            <div>source</div>
          </GenerateCollapsibleSection>
        </GenerationSectionPanel>
      </>
    );

    await expect.element(jumpButton()).not.toBeInTheDocument();
  });
});
