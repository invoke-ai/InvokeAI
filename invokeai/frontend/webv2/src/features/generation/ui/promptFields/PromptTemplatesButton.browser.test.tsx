import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { PromptTemplateCatalog } from '@features/generation/ui/usePromptTemplates';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { PromptTemplatesButton } from '@features/generation/ui/promptFields/PromptTemplatesButton';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const catalog: PromptTemplateCatalog = {
  create: vi.fn(),
  defaultTemplates: [],
  exportCsv: vi.fn(),
  importFile: vi.fn(),
  isLoaded: true,
  isLoading: false,
  personalTemplates: [],
  remove: vi.fn(),
  sharedTemplates: [],
  templates: [],
  update: vi.fn(),
};

const usePromptTemplates = vi.fn((_options?: { isEnabled?: boolean }) => catalog);
const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en' });

vi.mock('@features/generation/ui/usePromptTemplates', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  usePromptTemplates: (options?: { isEnabled?: boolean }) => usePromptTemplates(options),
}));

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({
    capabilities: { canManagePromptTemplates: false },
    notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn() },
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const applied: PromptTemplateSnapshot = {
  id: 'user-1',
  name: 'Cinematic',
  negativePrompt: '',
  positivePrompt: '{prompt}, cinematic',
};
const render = async (activeTemplate: PromptTemplateSnapshot | null) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Box aria-hidden bg="bg.muted" data-testid="popover-surface-style-probe" />
          <PromptTemplatesButton activeTemplate={activeTemplate} showSyntaxHighlighting={false} onApply={vi.fn()} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

const lastIsEnabled = (): boolean | undefined => usePromptTemplates.mock.calls.at(-1)?.[0]?.isEnabled;

beforeEach(() => {
  usePromptTemplates.mockClear();
  usePromptTemplates.mockReturnValue(catalog);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('the prompt templates button', () => {
  it('opens the catalog lazily on the standard popover surface', async () => {
    await render(null);

    expect(lastIsEnabled()).toBe(false);

    await act(async () => {
      await userEvent.click(host!.querySelector('button')!);
    });

    expect(lastIsEnabled()).toBe(true);
    const content = document.querySelector<HTMLElement>('[data-scope="popover"][data-part="content"]')!;
    const surface = getComputedStyle(
      host!.querySelector('[data-testid="popover-surface-style-probe"]')!
    ).backgroundColor;
    expect(getComputedStyle(content).backgroundColor).toBe(surface);
  });

  it('dims the name and says so when the applied template is gone', async () => {
    await render(applied);

    const name = [...host!.querySelectorAll('span')].find((span) => span.textContent === 'Cinematic');

    expect(lastIsEnabled()).toBe(true);
    expect(name).toBeTruthy();
    expect(getComputedStyle(name!).opacity).toBe('0.5');
  });

  it('is a labeled primary at rest, swapping the label for the applied name', async () => {
    // A labeled button rather than a bare icon: "Templates" until a template is
    // applied, then the applied template's name in its place.
    await render(null);
    const restLabel = [...host!.querySelectorAll('span')].find(
      (span) => span.textContent === 'widgets.generate.templatesButton'
    );

    expect(restLabel).toBeDefined();

    await act(() => root?.unmount());
    host?.remove();
    await render(applied);

    const name = [...host!.querySelectorAll('span')].find((span) => span.textContent === 'Cinematic');

    expect(name).toBeDefined();
    expect(
      [...host!.querySelectorAll('span')].find((span) => span.textContent === 'widgets.generate.templatesButton')
    ).toBeUndefined();
  });

  it('says nothing while the catalog is unread', async () => {
    usePromptTemplates.mockReturnValue({ ...catalog, isLoaded: false });
    await render(applied);

    const name = [...host!.querySelectorAll('span')].find((span) => span.textContent === 'Cinematic');

    expect(getComputedStyle(name!).opacity).toBe('1');
    usePromptTemplates.mockReturnValue(catalog);
  });
});
