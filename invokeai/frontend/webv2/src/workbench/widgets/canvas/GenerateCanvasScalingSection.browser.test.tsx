import { ChakraProvider } from '@chakra-ui/react';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

// A project with a Wan TI2V-5B model and a manual 1280x720 processing size. Stable identities, so only
// the capability table's arrival can change what the section computes.
const harness = vi.hoisted(() => ({
  commands: { widgets: { patchValues: () => undefined } },
  project: {
    canvas: { document: { bbox: { height: 720, width: 1280, x: 0, y: 0 } } },
    values: {
      canvas: { scaleMethod: 'manual', scaledHeight: 720, scaledWidth: 1280 },
      generate: {
        model: { base: 'wan', key: 'wan-ti2v', name: 'Wan 2.2 TI2V-5B', type: 'main', variant: 'ti2v_5b' },
        pidMode: 'off',
      },
    } as Record<string, Record<string, unknown>>,
  },
  sectionPreferences: { sectionsOpen: {}, setSectionOpen: () => undefined },
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: unknown) => unknown) => selector(harness.project),
  useWorkbenchCommands: () => harness.commands,
}));
vi.mock('@workbench/widgetState', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProjectWidgetValues: (project: typeof harness.project, widgetId: string) => project.values[widgetId] ?? {},
}));
vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useGenerationUi: () => ({ sectionPreferences: harness.sectionPreferences }),
}));

import { GenerateCanvasScalingSection } from './GenerateCanvasScalingSection';

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const render = async () => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '320px';
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerateCanvasScalingSection />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

/** The processing size the section's badge shows, as `[width, height]`. */
const shownSize = (): [number, number] | null => {
  const match = /(\d+)x(\d+)/.exec(host?.textContent ?? '');
  return match ? [Number(match[1]), Number(match[2])] : null;
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  resetArchitectureCapabilities();
});

describe('GenerateCanvasScalingSection', () => {
  it('shows the size the graph will generate at, on the grid the model variant declares', async () => {
    // Before the table the fallback grid is 8, and 720 sits on it.
    await render();
    expect(shownSize()).toEqual([1280, 720]);

    // TI2V-5B requires grid 32 rather than its base's 16; preserve variant and capability updates so shown and
    // compiled sizes agree.
    await act(() => {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    });

    const [width, height] = shownSize() ?? [0, 0];
    expect(width).toBe(1280);
    expect(height % 32).toBe(0);
    expect(height).not.toBe(720);
  });
});
