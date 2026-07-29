import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { PromptTemplateCatalog } from '@features/generation/ui/usePromptTemplates';

import { ChakraProvider } from '@chakra-ui/react';
import { promptTemplateKeys } from '@features/generation/data/promptTemplates';
import { PromptTemplateEditor } from '@features/generation/ui/promptFields/PromptTemplateEditor';
import { PromptTemplateImage } from '@features/generation/ui/promptFields/PromptTemplateImage';
import { PromptTemplatesPanel } from '@features/generation/ui/promptFields/PromptTemplatesPanel';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
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

const templateWithImage: PromptTemplateRecord = {
  hasImage: true,
  id: 'user-1',
  isDefault: false,
  isPublic: false,
  name: 'Cinematic',
  negativePrompt: '',
  positivePrompt: '{prompt}, cinematic',
  userId: 'user-1',
};

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
const IMAGE_FALLBACK = <span>fallback</span>;

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

const render = async (element: React.ReactNode, seed?: (queryClient: QueryClient) => void) => {
  host = document.createElement('div');
  host.style.width = '400px';
  document.body.append(host);
  root = createRoot(host);
  const queryClient = new QueryClient();
  seed?.(queryClient);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <QueryClientProvider client={queryClient}>
          <ChakraProvider value={system}>{element}</ChakraProvider>
        </QueryClientProvider>
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

    // The four snapshot fields only. Ownership and image metadata belong to the
    // catalog, not to persisted project state.
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
        isActiveTemplateMissing={false}
        catalog={createCatalog()}
        onApply={onApply}
        onCreate={vi.fn()}
        onDetach={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.click(buttonWithText('Stop using Cinematic'));
    });

    expect(onApply).toHaveBeenCalledWith(null);
  });

  // Deleting the applied template has to stop it applying — otherwise it keeps
  // shaping every prompt with no way left to reach it. But that is not the same
  // intent as choosing to clear it, so the panel stays open: the user is in the
  // middle of managing the list.
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
      await userEvent.click(host!.querySelector<HTMLButtonElement>('button[aria-label="Delete"]')!);
    });
    // The confirmation is portalled, so it is not under `host`.
    const dialog = document.querySelector('[role="alertdialog"]')!;
    const confirm = [...dialog.querySelectorAll('button')].find((button) => button.textContent === 'Delete')!;

    await act(async () => {
      await userEvent.click(confirm);
    });

    expect(onDetach).toHaveBeenCalled();
    expect(onApply).not.toHaveBeenCalled();
  });

  // Built-ins are shipped by the backend and rejected by it for non-admins, so
  // offering the controls would only produce a 403.
  it('offers edit and delete on your own templates but not on built-ins', async () => {
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

    const labels = [...host!.querySelectorAll('button')].map((button) => button.getAttribute('aria-label'));

    expect(labels.filter((label) => label === 'Edit')).toHaveLength(1);
    expect(labels.filter((label) => label === 'Delete')).toHaveLength(1);
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

  // The snapshot deliberately keeps applying after the template is gone, so a
  // queue item can still explain itself. Nothing said so, which left a prompt
  // being reshaped by something the list no longer contains.
  it('says when the applied template is no longer in the catalog', async () => {
    await render(
      <PromptTemplatesPanel
        activeTemplate={userTemplate}
        isActiveTemplateMissing
        catalog={createCatalog({ personalTemplates: [], templates: [sharedTemplate, defaultTemplate] })}
        onApply={vi.fn()}
        onCreate={vi.fn()}
        onDetach={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    expect(host!.textContent).toContain('was deleted');
    // And the way out is still offered beside it.
    expect(buttonWithText('Stop using Cinematic')).toBeTruthy();
  });

  it('says nothing of the sort while the template is still there', async () => {
    await render(
      <PromptTemplatesPanel
        activeTemplate={userTemplate}
        isActiveTemplateMissing={false}
        catalog={createCatalog()}
        onApply={vi.fn()}
        onCreate={vi.fn()}
        onDetach={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    expect(host!.textContent).not.toContain('was deleted');
  });

  it.each([
    ['twilight', 'Cinematic'],
    ['WATERMARK', 'Cinematic'],
    ['volumetric', 'Community'],
    ['FINGERS', 'Community'],
    ['bokeh', 'Photography'],
    ['LOW CONTRAST', 'Photography'],
  ])('finds case-insensitive prompt prose in the owning group (%s)', async (search, expectedName) => {
    const catalog = createCatalog({
      defaultTemplates: [defaultTemplate],
      personalTemplates: [{ ...userTemplate, negativePrompt: 'watermark', positivePrompt: '{prompt}, twilight' }],
      sharedTemplates: [
        { ...sharedTemplate, negativePrompt: 'extra fingers', positivePrompt: '{prompt}, volumetric rays' },
      ],
    });

    await render(
      <PromptTemplatesPanel
        activeTemplate={null}
        isActiveTemplateMissing={false}
        catalog={catalog}
        onApply={vi.fn()}
        onCreate={vi.fn()}
        onDetach={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.fill(host!.querySelector('input')!, search);
    });

    expect(host!.textContent).toContain(expectedName);
  });

  it('keeps name hits ahead of prose hits without moving templates between their groups', async () => {
    const catalog = createCatalog({
      defaultTemplates: [defaultTemplate],
      personalTemplates: [
        { ...userTemplate, name: 'Cinematic', positivePrompt: '{prompt}, bokeh' },
        { ...userTemplate, id: 'user-2', name: 'Bokeh study', positivePrompt: '{prompt}, portrait' },
      ],
      sharedTemplates: [{ ...sharedTemplate, positivePrompt: '{prompt}, bokeh' }],
    });

    await render(
      <PromptTemplatesPanel
        activeTemplate={null}
        isActiveTemplateMissing={false}
        catalog={catalog}
        onApply={vi.fn()}
        onCreate={vi.fn()}
        onDetach={vi.fn()}
        onEdit={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.fill(host!.querySelector('input')!, 'BOKEH');
    });

    const content = host!.textContent!;
    expect(content.indexOf('Your templates')).toBeLessThan(content.indexOf('Bokeh study'));
    expect(content.indexOf('Bokeh study')).toBeLessThan(content.indexOf('Cinematic'));
    expect(content.indexOf('Cinematic')).toBeLessThan(content.indexOf('Shared templates'));
    expect(content.indexOf('Shared templates')).toBeLessThan(content.indexOf('Community'));
    expect(content.indexOf('Community')).toBeLessThan(content.indexOf('Built-in templates'));
    expect(content.indexOf('Built-in templates')).toBeLessThan(content.indexOf('Photography'));
  });

  it('renders shared templates without edit or delete controls', async () => {
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

    expect(host!.textContent).toContain('Shared templates');
    expect(host!.textContent).toContain('Community');
    const labels = [...host!.querySelectorAll('button')].map((button) => button.getAttribute('aria-label'));
    expect(labels.filter((label) => label === 'Edit')).toHaveLength(1);
    expect(labels.filter((label) => label === 'Delete')).toHaveLength(1);
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

  it('preserves an existing image when an edit is saved immediately', async () => {
    const catalog = createCatalog();

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
      await userEvent.click(buttonWithText('Save'));
    });

    expect(catalog.update).toHaveBeenCalledWith(
      templateWithImage,
      expect.objectContaining({ image: { kind: 'preserve' } })
    );
  });

  it('marks an existing image for removal only after the user removes it', async () => {
    const catalog = createCatalog();

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
      await userEvent.click(buttonWithText('Save'));
    });

    expect(catalog.update).toHaveBeenCalledWith(
      templateWithImage,
      expect.objectContaining({ image: { kind: 'remove' } })
    );
  });

  it('marks a picked image as the replacement', async () => {
    const catalog = createCatalog();
    const replacement = new File(['new'], 'new.png', { type: 'image/png' });

    await render(
      <PromptTemplateEditor
        catalog={catalog}
        showSyntaxHighlighting
        template={templateWithImage}
        onCancel={vi.fn()}
        onSaved={vi.fn()}
      />
    );

    const input = host!.querySelector<HTMLInputElement>('input[type="file"]')!;
    await act(async () => {
      await userEvent.upload(input, replacement);
      await userEvent.click(buttonWithText('Save'));
    });

    expect(catalog.update).toHaveBeenCalledWith(
      templateWithImage,
      expect.objectContaining({ image: { blob: replacement, kind: 'replace' } })
    );
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

it('revokes a fetched template image blob URL when it unmounts', async () => {
  const createObjectUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:template-preview');
  const revokeObjectUrl = vi.spyOn(URL, 'revokeObjectURL');

  await render(<PromptTemplateImage alt="" fallback={IMAGE_FALLBACK} template={templateWithImage} />, (queryClient) =>
    queryClient.setQueryData(promptTemplateKeys.image(templateWithImage.id), new Blob(['image']))
  );

  expect(createObjectUrl).toHaveBeenCalledOnce();
  await act(() => root?.unmount());
  root = null;
  expect(revokeObjectUrl).toHaveBeenCalledWith('blob:template-preview');

  createObjectUrl.mockRestore();
  revokeObjectUrl.mockRestore();
});

describe('prompt template image outlines', () => {
  const modes = [
    ['light', 'light', 'oklch(0 0 0 / 0.1)'],
    ['dark', 'classic', 'oklch(1 0 0 / 0.1)'],
  ] as const;

  for (const [mode, theme, outlineColor] of modes) {
    it(`gives local preview images a one-pixel inset outline in ${mode} mode`, async () => {
      document.documentElement.className = mode;
      document.documentElement.dataset.theme = theme;

      await render(
        <PromptTemplateImage
          alt="Local preview"
          fallback={IMAGE_FALLBACK}
          localPreviewUrl="data:image/png;base64,iVBORw0KGgo="
          template={templateWithImage}
        />
      );

      const image = host!.querySelector<HTMLImageElement>('img[alt="Local preview"]')!;

      expect(getComputedStyle(image).outlineStyle).toBe('solid');
      expect(getComputedStyle(image).outlineWidth).toBe('1px');
      expect(getComputedStyle(image).outlineOffset).toBe('-1px');
      expect(getComputedStyle(image).outlineColor).toBe(outlineColor);
    });
  }

  it('allows callers to override the default outline', async () => {
    await render(
      <PromptTemplateImage
        alt="Overridden preview"
        fallback={IMAGE_FALLBACK}
        localPreviewUrl="data:image/png;base64,iVBORw0KGgo="
        outline="2px dotted red"
        outlineOffset="0"
        template={templateWithImage}
      />
    );

    const image = host!.querySelector<HTMLImageElement>('img[alt="Overridden preview"]')!;
    expect(getComputedStyle(image).outline).toBe('rgb(255, 0, 0) dotted 2px');
    expect(getComputedStyle(image).outlineOffset).toBe('0px');
  });
});
