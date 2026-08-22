import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { PromptTemplateCatalog } from '@features/generation/ui/usePromptTemplates';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { exportPromptTemplates } from '@features/generation/data/promptTemplates';
import { expectRowInteractionsToMatch } from '@features/generation/ui/promptFields/promptFieldsBrowserTestUtils';
import { PromptTemplateEditor } from '@features/generation/ui/promptFields/PromptTemplateEditor';
import { PromptTemplatesPanel } from '@features/generation/ui/promptFields/PromptTemplatesPanel';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { Row } from '@platform/ui/Row';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const dependencies = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  canManagePromptTemplates: false,
  downloadBlob: vi.fn(),
}));

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({
    capabilities: { canManagePromptTemplates: dependencies.canManagePromptTemplates },
    notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn() },
  }),
}));

vi.mock('@platform/browser/downloadBlob', () => ({ downloadBlob: dependencies.downloadBlob }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  apiFetch: dependencies.apiFetch,
}));

const translation = {
  common: {
    cancel: 'Cancel',
    delete: 'Delete',
    edit: 'Edit',
    negative: 'Negative',
    prompt: 'Prompt',
    save: 'Save',
  },
  widgets: {
    generate: {
      promptTemplates: {
        title: 'Prompt templates',
        newTemplate: 'New template',
        editTemplate: 'Edit template',
        search: 'Search templates',
        yourTemplates: 'Your templates',
        sharedTemplates: 'Shared templates',
        defaultTemplates: 'Built-in templates',
        noTemplatesYet: 'No templates yet. Create one to reuse a prompt with {prompt} where your own text goes.',
        noMatches: 'No matching templates',
        applied: 'Template: {{name}}',
        appliedMissing: 'Template: {{name}} (deleted)',
        appliedMissingHelp: '“{{name}}” was deleted. It still shapes your prompt until you stop using it.',
        clearApplied: 'Stop using {{name}}',
        clear: 'Clear',
        name: 'Name',
        namePlaceholder: 'Cinematic',
        nameHelp: 'Templates you create are saved to your account.',
        nameTooLong: 'Name must be at most 128 characters',
        positivePrompt: 'Template prompt',
        positivePromptPlaceholder: '{prompt}. cinematic still, 35mm, shallow depth of field',
        negativePrompt: 'Template negative prompt',
        negativePromptPlaceholder: '{prompt}, blurry, lowres',
        resizePositivePrompt: 'Resize template prompt',
        resizeNegativePrompt: 'Resize template negative prompt',
        insertPlaceholderHelp:
          'Insert {prompt}, which your own prompt replaces. Without it the template is added to the end.',
        placeholderAlreadyUsed: 'Only the first {prompt} is replaced',
        image: 'Preview image',
        addImage: 'Add an image',
        replaceImage: 'Replace image',
        removeImage: 'Remove image',
        import: 'Import',
        importHelp: 'Import templates from a CSV or JSON file with name, prompt and negative_prompt.',
        export: 'Export',
        exportHelp: 'Download every user template as a CSV file.',
        couldNotSave: 'Could not save this template',
        couldNotDelete: 'Could not delete this template',
        couldNotImport: 'Could not import templates',
        couldNotExport: 'Could not export templates',
        deleteTitle: 'Delete template',
        deleteBody: 'Delete “{{name}}”? This cannot be undone.',
        viewMerged: 'Show the merged prompt',
        editAuthored: 'Back to editing your prompt',
        loading: 'Loading templates…',
      },
    },
  },
};

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation } } });

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const userTemplate: PromptTemplateRecord = {
  hasImage: false,
  id: 'user-1',
  isDefault: false,
  isPublic: false,
  name: 'Cinematic',
  negativePrompt: '',
  positivePrompt: '{prompt}, cinematic still',
  userId: 'user-1',
};

const sharedTemplate: PromptTemplateRecord = {
  hasImage: false,
  id: 'shared-1',
  isDefault: false,
  isPublic: true,
  name: 'Community',
  negativePrompt: '',
  positivePrompt: '{prompt}, shared',
  userId: 'user-2',
};

const defaultTemplate: PromptTemplateRecord = {
  hasImage: false,
  id: 'default-1',
  isDefault: true,
  isPublic: false,
  name: 'Photography',
  negativePrompt: 'low contrast',
  positivePrompt: '{prompt}. photography, bokeh',
  userId: 'system',
};
const createCatalog = (overrides: Partial<PromptTemplateCatalog> = {}): PromptTemplateCatalog => ({
  create: vi.fn(),
  defaultTemplates: [defaultTemplate],
  exportCsv: vi.fn(),
  importFile: vi.fn(),
  isLoaded: true,
  isLoading: false,
  personalTemplates: [userTemplate],
  remove: vi.fn(),
  sharedTemplates: [sharedTemplate],
  templates: [userTemplate, sharedTemplate, defaultTemplate],
  update: vi.fn(),
  ...overrides,
});

const StatefulPromptTemplatesPanel = ({ catalog }: { catalog: PromptTemplateCatalog }) => {
  const [activeTemplate, setActiveTemplate] = useState<PromptTemplateSnapshot | null>(null);

  return (
    <>
      <Box aria-hidden bg="bg.muted/60" data-testid="row-hover-style-probe" />
      <Row asChild>
        <button aria-label="Row probe" type="button">
          Row probe
        </button>
      </Row>
      <PromptTemplatesPanel
        activeTemplate={activeTemplate}
        catalog={catalog}
        isActiveTemplateMissing={false}
        onApply={setActiveTemplate}
        onCreate={vi.fn()}
        onDetach={vi.fn()}
        onEdit={vi.fn()}
      />
    </>
  );
};

const render = async (element: React.ReactNode) => {
  host = document.createElement('div');
  host.style.width = '400px';
  document.body.append(host);
  root = createRoot(host);
  const queryClient = new QueryClient();

  const contents = (
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>{element}</ChakraProvider>
      </QueryClientProvider>
    </I18nextProvider>
  );

  await act(() => {
    root?.render(contents);
  });
};

const buttonWithText = (text: string): HTMLButtonElement =>
  [...host!.querySelectorAll('button')].find((button) => button.textContent?.includes(text))!;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  vi.restoreAllMocks();
  dependencies.apiFetch.mockReset();
  dependencies.canManagePromptTemplates = false;
  dependencies.downloadBlob.mockReset();
  accountLifecycle.invalidate();
  document.documentElement.className = '';
  delete document.documentElement.dataset.theme;
});

describe('the prompt templates panel', () => {
  it('applies the template that was clicked', async () => {
    const onApply = vi.fn();

    await render(
      <PromptTemplatesPanel
        activeTemplate={null}
        isActiveTemplateMissing={false}
        catalog={createCatalog()}
        onApply={onApply}
        onCreate={vi.fn()}
        onDetach={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.click(buttonWithText('Cinematic'));
    });

    expect(onApply).toHaveBeenCalledWith({
      id: userTemplate.id,
      name: userTemplate.name,
      negativePrompt: userTemplate.negativePrompt,
      positivePrompt: userTemplate.positivePrompt,
    });
  });

  it('uses the shared Row interaction contract for inactive templates and keeps management controls separate', async () => {
    await render(<StatefulPromptTemplatesPanel catalog={createCatalog()} />);

    const probe = host!.querySelector<HTMLButtonElement>('button[aria-label="Row probe"]')!;
    const row = buttonWithText('Cinematic');
    const hoverBackgroundColor = getComputedStyle(
      host!.querySelector('[data-testid="row-hover-style-probe"]')!
    ).backgroundColor;
    const edit = host!.querySelector<HTMLButtonElement>('button[aria-label="Edit: Cinematic"]')!;
    const remove = host!.querySelector<HTMLButtonElement>('button[aria-label="Delete: Cinematic"]')!;

    expect(row.getAttribute('aria-current')).toBeNull();
    expect(row.contains(edit)).toBe(false);
    expect(row.contains(remove)).toBe(false);

    await expectRowInteractionsToMatch(probe, row, hoverBackgroundColor);
  });

  it('detaches rather than clears when the applied template is deleted', async () => {
    const onApply = vi.fn();
    const onDetach = vi.fn();

    await render(
      <PromptTemplatesPanel
        activeTemplate={userTemplate}
        isActiveTemplateMissing={false}
        catalog={createCatalog()}
        onApply={onApply}
        onCreate={vi.fn()}
        onDetach={onDetach}
        onEdit={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.click(host!.querySelector<HTMLButtonElement>('button[aria-label="Delete: Cinematic"]')!);
    });
    const dialog = document.querySelector('[role="alertdialog"]')!;
    const confirm = [...dialog.querySelectorAll('button')].find((button) => button.textContent === 'Delete')!;

    await act(async () => {
      await userEvent.click(confirm);
    });

    expect(onDetach).toHaveBeenCalled();
    expect(onApply).not.toHaveBeenCalled();
  });

  it('finds fuzzy template names', async () => {
    await render(
      <PromptTemplatesPanel
        activeTemplate={null}
        isActiveTemplateMissing={false}
        catalog={createCatalog()}
        onApply={vi.fn()}
        onCreate={vi.fn()}
        onDetach={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.fill(host!.querySelector('input')!, 'cnmt');
    });

    expect(host!.textContent).toContain('Cinematic');
    expect(host!.textContent).not.toContain('Photography');
  });
});

describe('the prompt template editor', () => {
  it('saves a new template as the user’s own', async () => {
    const catalog = createCatalog();

    await render(
      <PromptTemplateEditor
        catalog={catalog}
        showSyntaxHighlighting
        template={null}
        onCancel={vi.fn()}
        onSaved={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.fill(host!.querySelector('input')!, 'Watercolour');
    });
    await act(async () => {
      await userEvent.fill(host!.querySelectorAll('textarea')[0]!, '{prompt}, watercolour');
    });
    await act(async () => {
      await userEvent.click(buttonWithText('Save'));
    });

    expect(catalog.create).toHaveBeenCalledWith({
      image: null,
      name: 'Watercolour',
      negativePrompt: '',
      positivePrompt: '{prompt}, watercolour',
    });
  });
});

it('does not download a prompt-template export whose body finishes after account rotation', async () => {
  let resolveBody!: (blob: Blob) => void;
  const body = new Promise<Blob>((resolve) => {
    resolveBody = resolve;
  });
  const owner = accountLifecycle.activate('prompt-export-a', ':user:prompt-export-a');

  dependencies.canManagePromptTemplates = true;
  dependencies.apiFetch.mockResolvedValue({ blob: () => body });

  await render(
    <PromptTemplatesPanel
      activeTemplate={null}
      catalog={createCatalog({ exportCsv: exportPromptTemplates })}
      isActiveTemplateMissing={false}
      onApply={vi.fn()}
      onCreate={vi.fn()}
      onDetach={vi.fn()}
      onEdit={vi.fn()}
    />
  );

  await act(async () => {
    await userEvent.click(buttonWithText('Export'));
    await vi.waitFor(() => expect(dependencies.apiFetch).toHaveBeenCalledOnce());
  });

  expect(dependencies.apiFetch).toHaveBeenCalledWith('/api/v1/style_presets/export', { signal: owner.signal });

  await act(async () => {
    accountLifecycle.activate('prompt-export-b', ':user:prompt-export-b');
    resolveBody(new Blob(['name,prompt']));
    await body;
  });

  expect(dependencies.downloadBlob).not.toHaveBeenCalled();
});
