/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider, Stack } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GenerateCollapsibleSection } from './GenerateCollapsibleSection';

vi.mock('@features/generation/ui/GenerationUiContext', () => ({
  useGenerationUi: () => ({
    sectionPreferences: { sectionsOpen: { 'open-section': true, 'shut-section': false }, setSectionOpen: vi.fn() },
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const render = async (children: React.ReactNode) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(<ChakraProvider value={system}>{children}</ChakraProvider>);
  });
};

const getSections = () => [...document.querySelectorAll<HTMLElement>('.generate-section')];

describe('GenerateCollapsibleSection list styling', () => {
  it('styles by DOM state so uncontrolled sections (no sectionId) still read as cards', async () => {
    // Upscale's sections pass only defaultOpen — openness never reaches the
    // component as a prop, so any prop-driven styling is blind to them.
    await render(
      <GenerateCollapsibleSection defaultOpen label="Uncontrolled">
        <div>body</div>
      </GenerateCollapsibleSection>
    );

    const [section] = getSections();

    expect(section?.dataset.state).toBe('open');
    expect(getComputedStyle(section!).backgroundColor).not.toBe('rgba(0, 0, 0, 0)');
    // A lone row has no neighboring section, so no separator shows.
    expect(getComputedStyle(section!, '::before').opacity).toBe('0');
    expect(getComputedStyle(section!, '::after').content).toBe('none');
  });

  it('draws one overlay hairline per section|section boundary and fades the ones touching an open card', async () => {
    await render(
      <Stack gap={1}>
        <GenerateCollapsibleSection label="A" sectionId="open-section">
          <div>a</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection label="B" sectionId="shut-section">
          <div>b</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection label="C" sectionId="shut-section">
          <div>c</div>
        </GenerateCollapsibleSection>
      </Stack>
    );

    const [open, afterOpen, afterClosed] = getSections();

    expect(open?.dataset.state).toBe('open');
    expect(afterOpen?.dataset.state).toBe('closed');

    // The separators are pseudo-element overlays centered in the gap — real
    // borders or margins would move content on open/close. Each boundary gets
    // exactly ONE 1px line (the lower row's ::before): a second overlapping
    // half would stack the translucent border color into a heavier line.
    for (const section of [open, afterOpen, afterClosed]) {
      expect(getComputedStyle(section!).borderTopWidth).toBe('0px');
      expect(getComputedStyle(section!).marginTop).toBe('0px');
      expect(getComputedStyle(section!, '::before').position).toBe('absolute');
      expect(getComputedStyle(section!, '::before').top).toBe('-2px');
      expect(getComputedStyle(section!, '::before').borderTopWidth).toBe('1px');
      // Separators live only between rows: nothing ever closes the list.
      expect(getComputedStyle(section!, '::after').content).toBe('none');
    }

    // The lines touching the open card are faded: its own, and the one under
    // it (owned by the next row); the closed|closed boundary stays visible.
    expect(getComputedStyle(open!, '::before').opacity).toBe('0');
    expect(getComputedStyle(afterOpen!, '::before').opacity).toBe('0');
    expect(getComputedStyle(afterClosed!, '::before').opacity).toBe('1');
    // Closed rows never carry the card treatment.
    expect(getComputedStyle(afterOpen!).backgroundColor).toBe('rgba(0, 0, 0, 0)');
  });

  it('shows separators only between sections — none above the first row, even after a non-section', async () => {
    await render(
      <Stack gap={1}>
        <div>model card stand-in</div>
        <GenerateCollapsibleSection label="A" sectionId="shut-section">
          <div>a</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection label="B" sectionId="shut-section">
          <div>b</div>
        </GenerateCollapsibleSection>
        <GenerateCollapsibleSection label="C" sectionId="open-section">
          <div>c</div>
        </GenerateCollapsibleSection>
      </Stack>
    );

    const [first, second, open] = getSections();

    // A closed first row draws nothing above itself: the element before it is
    // not a section, so that boundary is not section|section.
    expect(getComputedStyle(first!, '::before').opacity).toBe('0');
    expect(getComputedStyle(second!, '::before').opacity).toBe('1');
    // The boundary above the open card is the card's own ::before — faded.
    expect(getComputedStyle(open!, '::before').opacity).toBe('0');
  });
});
