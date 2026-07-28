import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { PromptTemplateCatalog } from '@features/generation/ui/usePromptTemplates';

import { ChakraProvider } from '@chakra-ui/react';
import { fetchPromptTemplateImage } from '@features/generation/data/promptTemplates';
import { PromptTemplateEditor } from '@features/generation/ui/promptFields/PromptTemplateEditor';
import { PromptTemplatesPanel } from '@features/generation/ui/promptFields/PromptTemplatesPanel';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

// Both panels report failures through Generation's UI port and read the
// import/export capability from it; only the app composes that. The rest is real.
vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({
    capabilities: { canManagePromptTemplates: false },
    notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn() },
  }),
}));

// The editor loads an existing preview image back so saving does not drop it.
// Stubbed so a template with an image is driven from the test rather than the
// network, and so a regression shows up as a failed assertion.
vi.mock('@features/generation/data/promptTemplates', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  fetchPromptTemplateImage: vi.fn().mockResolvedValue(null),
}));

// The subtree of `public/locales/en.json` these panels read, copied verbatim so
// the assertions below can name the strings a user actually sees. Importing the
// catalogue itself is not an option — Vite does not serve `public/` to modules.
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
        defaultTemplates: 'Built-in templates',
        noTemplatesYet: 'No templates yet. Create one to reuse a prompt with {prompt} where your own text goes.',
        noMatches: 'No matching templates',
        applied: 'Template: {{name}}',
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

const templateWithImage: PromptTemplateRecord = {
  id: 'user-1',
  imageUrl: 'http://host/image.png',
  isDefault: false,
  name: 'Cinematic',
  negativePrompt: '',
  positivePrompt: '{prompt}, cinematic',
};

const userTemplate: PromptTemplateRecord = {
  id: 'user-1',
  imageUrl: null,
  isDefault: false,
  name: 'Cinematic',
  negativePrompt: '',
  positivePrompt: '{prompt}, cinematic still',
};

const defaultTemplate: PromptTemplateRecord = {
  id: 'default-1',
  imageUrl: null,
  isDefault: true,
  name: 'Photography',
  negativePrompt: '',
  positivePrompt: '{prompt}. photography, bokeh',
};

const createCatalog = (overrides: Partial<PromptTemplateCatalog> = {}): PromptTemplateCatalog => ({
  create: vi.fn(),
  defaultTemplates: [defaultTemplate],
  exportCsv: vi.fn(),
  importFile: vi.fn(),
  isLoading: false,
  remove: vi.fn(),
  templates: [userTemplate, defaultTemplate],
  update: vi.fn(),
  userTemplates: [userTemplate],
  ...overrides,
});

const render = async (element: React.ReactNode) => {
  host = document.createElement('div');
  host.style.width = '400px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>{element}</ChakraProvider>
      </I18nextProvider>
    );
  });
};

const buttonWithText = (text: string): HTMLButtonElement =>
  [...host!.querySelectorAll('button')].find((button) => button.textContent?.includes(text))!;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('the prompt templates panel', () => {
  it('applies the template that was clicked', async () => {
    const onApply = vi.fn();

    await render(
      <PromptTemplatesPanel
        activeTemplate={null}
        catalog={createCatalog()}
        onApply={onApply}
        onCreate={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.click(buttonWithText('Cinematic'));
    });

    // The four snapshot fields only. `isDefault` and the absolute `imageUrl`
    // belong to the catalog, not to persisted project state.
    expect(onApply).toHaveBeenCalledWith({
      id: userTemplate.id,
      name: userTemplate.name,
      negativePrompt: userTemplate.negativePrompt,
      positivePrompt: userTemplate.positivePrompt,
    });
  });

  it('clears the applied template', async () => {
    const onApply = vi.fn();

    await render(
      <PromptTemplatesPanel
        activeTemplate={userTemplate}
        catalog={createCatalog()}
        onApply={onApply}
        onCreate={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.click(buttonWithText('Stop using Cinematic'));
    });

    expect(onApply).toHaveBeenCalledWith(null);
  });

  // Built-ins are shipped by the backend and rejected by it for non-admins, so
  // offering the controls would only produce a 403.
  it('offers edit and delete on your own templates but not on built-ins', async () => {
    await render(
      <PromptTemplatesPanel
        activeTemplate={null}
        catalog={createCatalog()}
        onApply={vi.fn()}
        onCreate={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    const labels = [...host!.querySelectorAll('button')].map((button) => button.getAttribute('aria-label'));

    expect(labels.filter((label) => label === 'Edit')).toHaveLength(1);
    expect(labels.filter((label) => label === 'Delete')).toHaveLength(1);
  });

  it('searches prompt text, not just the name', async () => {
    await render(
      <PromptTemplatesPanel
        activeTemplate={null}
        catalog={createCatalog()}
        onApply={vi.fn()}
        onCreate={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.fill(host!.querySelector('input')!, 'bokeh');
    });

    expect(host!.textContent).toContain('Photography');
    expect(host!.textContent).not.toContain('Cinematic');
  });
});

describe('the prompt template editor', () => {
  it('appends the placeholder and then refuses to add a second one', async () => {
    await render(
      <PromptTemplateEditor
        catalog={createCatalog()}
        showSyntaxHighlighting
        template={null}
        onCancel={vi.fn()}
        onSaved={vi.fn()}
      />
    );

    const insertPlaceholder = buttonWithText('{prompt}');

    await act(async () => {
      await userEvent.click(insertPlaceholder);
    });

    expect(host!.querySelectorAll('textarea')[0]!.value).toBe('{prompt}');
    expect(buttonWithText('{prompt}').disabled).toBe(true);
  });

  // Authoring always writes a private, user-owned template; the backend's
  // `default` type and `is_public` flag are not exposed here.
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

  // Regression: the existing image is fetched on mount so that saving does not
  // drop it, and `image: null` was read as "still loading". Removing it before
  // that landed therefore looked identical to not having loaded yet, and the
  // arriving blob put the image the user had just taken off straight back.
  it('does not resurrect an image removed while it was still loading', async () => {
    const catalog = createCatalog();
    let resolveImage: (image: Blob) => void = () => {};

    vi.mocked(fetchPromptTemplateImage).mockReturnValueOnce(
      new Promise<Blob | null>((resolve) => {
        resolveImage = resolve;
      })
    );

    await render(
      <PromptTemplateEditor
        catalog={catalog}
        showSyntaxHighlighting
        template={templateWithImage}
        onCancel={vi.fn()}
        onSaved={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.click(host!.querySelector<HTMLButtonElement>('button[aria-label="Remove image"]')!);
    });

    await act(async () => {
      resolveImage(new Blob(['old'], { type: 'image/png' }));
      await Promise.resolve();
    });

    await act(async () => {
      await userEvent.click(buttonWithText('Save'));
    });

    expect(catalog.update).toHaveBeenCalledWith('user-1', expect.objectContaining({ image: null }));
  });

  it('keeps Save out of reach until the template is named', async () => {
    await render(
      <PromptTemplateEditor
        catalog={createCatalog()}
        showSyntaxHighlighting
        template={null}
        onCancel={vi.fn()}
        onSaved={vi.fn()}
      />
    );

    expect(buttonWithText('Save').disabled).toBe(true);

    await act(async () => {
      await userEvent.fill(host!.querySelector('input')!, 'Named');
    });

    expect(buttonWithText('Save').disabled).toBe(false);
  });
});
