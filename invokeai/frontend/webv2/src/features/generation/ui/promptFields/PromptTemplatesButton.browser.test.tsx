import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { PromptTemplateCatalog } from '@features/generation/ui/usePromptTemplates';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { waitForComputedStyles } from '@features/generation/ui/promptFields/promptFieldsBrowserTestUtils';
import { PromptTemplatesButton } from '@features/generation/ui/promptFields/PromptTemplatesButton';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
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

// The real hook is a react-query observer; what matters here is the argument it
// is handed, which decides whether the request goes out at all.
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

const translation = {
  common: { cancel: 'Cancel', delete: 'Delete', edit: 'Edit', save: 'Save' },
  widgets: {
    generate: {
      promptTemplates: {
        applied: 'Template: {{name}}',
        appliedMissing: 'Template: {{name}} (deleted)',
        appliedMissingHelp: '“{{name}}” was deleted. It still shapes your prompt until you stop using it.',
        clearApplied: 'Stop using {{name}}',
        defaultTemplates: 'Built-in templates',
        newTemplate: 'New template',
        noTemplatesYet: 'No templates yet.',
        search: 'Search templates',
        title: 'Prompt templates',
        yourTemplates: 'Your templates',
      },
    },
  },
};

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation } } });

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const applied: PromptTemplateSnapshot = {
  id: 'user-1',
  name: 'Cinematic',
  negativePrompt: '',
  positivePrompt: '{prompt}, cinematic',
};
const template: PromptTemplateRecord = {
  ...applied,
  hasImage: false,
  isDefault: false,
  isPublic: false,
  userId: 'user-1',
};

const render = async (activeTemplate: PromptTemplateSnapshot | null) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <Box aria-hidden bg="bg.muted" data-testid="row-hover-style-probe" />
            <PromptTemplatesButton activeTemplate={activeTemplate} showSyntaxHighlighting={false} onApply={vi.fn()} />
          </ChakraProvider>
        </I18nextProvider>
      </QueryClientProvider>
    );
  });
};

/** What the hook was handed on the most recent render. */
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
  // react-query enables a query if any one observer wants it, and this observer
  // always did — so the widget's own gating did nothing at all while this button
  // was mounted, which is always.
  it('does not ask for the catalog with nothing applied and the list closed', async () => {
    await render(null);

    expect(lastIsEnabled()).toBe(false);
  });

  it('asks for it once the list is opened', async () => {
    await render(null);

    await act(async () => {
      await userEvent.click(host!.querySelector('button')!);
    });

    expect(lastIsEnabled()).toBe(true);
  });

  it('keeps the template row hover visible against the popover surface', async () => {
    usePromptTemplates.mockReturnValue({
      ...catalog,
      personalTemplates: [template],
      templates: [template],
    });
    await render(null);
    const hoverBackgroundColor = getComputedStyle(
      host!.querySelector('[data-testid="row-hover-style-probe"]')!
    ).backgroundColor;

    await act(async () => {
      await userEvent.click(host!.querySelector<HTMLButtonElement>('button[aria-label="Prompt templates"]')!);
    });

    const row = [...document.querySelectorAll<HTMLButtonElement>('button')].find((button) =>
      button.textContent?.includes('Cinematic')
    )!;
    const popover = row.closest<HTMLElement>('[data-scope="popover"][data-part="content"]')!;

    await act(async () => {
      await userEvent.hover(row);
    });
    await waitForComputedStyles(row, { backgroundColor: hoverBackgroundColor });

    expect(getComputedStyle(row).backgroundColor).not.toBe(getComputedStyle(popover).backgroundColor);
  });

  // An applied template has to be re-read whether the list is open or not: to
  // refresh its text after an edit, and to notice it has been deleted.
  it('asks for it whenever a template is applied', async () => {
    await render(applied);

    expect(lastIsEnabled()).toBe(true);
  });

  it('dims the name and says so when the applied template is gone', async () => {
    await render(applied);

    const name = [...host!.querySelectorAll('span')].find((span) => span.textContent === 'Cinematic');

    expect(name).toBeTruthy();
    expect(getComputedStyle(name!).opacity).toBe('0.5');
  });

  it('leaves the name alone while the template is still there', async () => {
    usePromptTemplates.mockReturnValue({
      ...catalog,
      templates: [
        {
          ...applied,
          hasImage: false,
          isDefault: false,
          isPublic: false,
          userId: 'user-1',
        },
      ],
    });
    await render(applied);

    const name = [...host!.querySelectorAll('span')].find((span) => span.textContent === 'Cinematic');

    expect(getComputedStyle(name!).opacity).toBe('1');
    usePromptTemplates.mockReturnValue(catalog);
  });

  // A catalog that has not been read cannot say anything is missing — an
  // unfinished or failed fetch must not read as a deletion.
  it('says nothing while the catalog is unread', async () => {
    usePromptTemplates.mockReturnValue({ ...catalog, isLoaded: false });
    await render(applied);

    const name = [...host!.querySelectorAll('span')].find((span) => span.textContent === 'Cinematic');

    expect(getComputedStyle(name!).opacity).toBe('1');
    usePromptTemplates.mockReturnValue(catalog);
  });
});
