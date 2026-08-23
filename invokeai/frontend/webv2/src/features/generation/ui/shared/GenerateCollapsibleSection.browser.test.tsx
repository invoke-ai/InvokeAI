import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GenerateCollapsibleSection } from './GenerateCollapsibleSection';

vi.mock('@features/generation/ui/GenerationUiContext', () => ({
  useGenerationUi: () => ({
    sectionPreferences: { sectionsOpen: {}, setSectionOpen: vi.fn() },
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

describe('GenerateCollapsibleSection', () => {
  it('styles by DOM state so uncontrolled sections (no sectionId) still read as cards', async () => {
    // Upscale's sections pass only defaultOpen — openness never reaches the
    // component as a prop, so any prop-driven styling is blind to them.
    await render(
      <GenerateCollapsibleSection defaultOpen label="Uncontrolled">
        <div>body</div>
      </GenerateCollapsibleSection>
    );

    const section = document.querySelector<HTMLElement>('.generate-section');

    expect(section?.dataset.state).toBe('open');
    expect(getComputedStyle(section!).backgroundColor).not.toBe('rgba(0, 0, 0, 0)');
  });
});
