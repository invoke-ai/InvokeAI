import type { GalleryImage } from '@features/gallery';
import type { ModelConfig } from '@features/models/core/types';
import type { WorkflowLibraryBrowseSnapshot, WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { auditAccessibility, contrastOffenderTexts } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { takeNewProjectRequest } from '@workbench/projects/newProjectRequest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const pending = vi.hoisted(() => () => new Promise<never>(() => {}));

const harness = vi.hoisted(() => ({
  canManageModels: true,
  /** What the clipboard holds for the next read. */
  clipboard: null as Blob | null,
  importFile: vi.fn(),
  navigate: vi.fn(),
  patchPreferences: vi.fn(() => Promise.resolve()),
  preferences: {
    enableModelDescriptions: true,
    launchpadNewProjectIntent: 'generate',
    launchpadNewProjectModelKey: null as string | null,
  },
  upload: vi.fn(),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values
        ? `${key}:${Object.entries(values)
            .map(([name, value]) => `${name}=${String(value)}`)
            .join(',')}`
        : key,
  }),
}));
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  Link: ({ children, search: _search, to, ...props }: { children?: unknown; search?: unknown; to: string }) => (
    <a href={to} {...props}>
      {children as never}
    </a>
  ),
  useNavigate: () => harness.navigate,
}));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useCapabilities: () => ({ canManageModels: harness.canManageModels }),
}));
vi.mock('@workbench/projects/useProjectFileActions', () => ({
  useImportProjectFile: () => harness.importFile,
}));
vi.mock('@workbench/settings/store', () => ({
  getWorkbenchPreferences: () => harness.preferences,
  patchWorkbenchPreferences: harness.patchPreferences,
  useWorkbenchPreferenceSelector: (select: (preferences: typeof harness.preferences) => unknown) =>
    select(harness.preferences),
}));
// The served capability table decides grids and optimal sizes; answer with the pinned fixture instead of the network.
vi.mock('@features/generation/data/architectureCapabilitiesApi', async () => {
  const { default: rows } = await import('@features/generation/core/__fixtures__/architectureCapabilities.json');

  return { getArchitectureCapabilities: () => Promise.resolve(rows) };
});
// The clipboard and the upload are the two seams of the image start; the slot's own picker is covered by the gallery.
vi.mock('@platform/browser/clipboardImage', () => ({
  readClipboardImage: () => Promise.resolve(harness.clipboard),
}));
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  uploadGalleryImage: harness.upload,
}));
// The model picker pins related models once it opens; there are none here.
vi.mock('@features/models/data/relationshipsApi', () => ({
  addModelRelationship: pending,
  getRelatedModelKeys: () => Promise.resolve([]),
  removeModelRelationship: pending,
}));

// Drive a real external browse store so the picker's subscriptions and selectors run, while owning its contents.
const browse = vi.hoisted(() => ({
  ensureWorkflowLibraryBrowseLoaded: vi.fn(() => Promise.resolve()),
  getSnapshot: null as null | (() => WorkflowLibraryBrowseSnapshot),
  loadNextWorkflowLibraryPage: vi.fn(),
  setSnapshot: null as null | ((snapshot: WorkflowLibraryBrowseSnapshot) => void),
  setWorkflowLibraryBrowseFilter: vi.fn(),
}));

const EMPTY_BROWSE: WorkflowLibraryBrowseSnapshot = {
  entries: [],
  error: null,
  filter: { category: 'user', search: '', tag: null },
  page: 0,
  pages: 0,
  status: 'idle',
  tagCounts: [],
  total: 0,
  userTotal: null,
};

vi.mock('@features/workflow/data/libraryBrowseStore', async () => {
  const { createExternalStore } = await import('@platform/state/externalStore');
  const store = createExternalStore<WorkflowLibraryBrowseSnapshot>(EMPTY_BROWSE);

  browse.getSnapshot = store.getSnapshot;
  browse.setSnapshot = store.setSnapshot;

  return {
    ensureWorkflowLibraryBrowseLoaded: browse.ensureWorkflowLibraryBrowseLoaded,
    getWorkflowLibraryBrowseSnapshot: store.getSnapshot,
    loadNextWorkflowLibraryPage: browse.loadNextWorkflowLibraryPage,
    setWorkflowLibraryBrowseFilter: browse.setWorkflowLibraryBrowseFilter,
    useWorkflowLibraryBrowseSelector: store.useSelector,
  };
});

const { NewProjectButton } = await import('./NewProjectButton');
// The button loads the dialog lazily; evaluate it now so the mocked stores above exist before the first open.
await import('./NewProjectDialog');

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const model = (overrides: Partial<ModelConfig> & Pick<ModelConfig, 'key' | 'name'>): ModelConfig => ({
  base: 'sdxl',
  default_settings: null,
  description: null,
  file_size: 1024,
  format: 'checkpoint',
  hash: `hash-${overrides.key}`,
  path: `/models/${overrides.key}.safetensors`,
  source: `/models/${overrides.key}.safetensors`,
  source_type: 'path',
  trigger_phrases: [],
  type: 'main',
  ...overrides,
});

const CATALOG: ModelConfig[] = [
  model({ default_settings: { height: 1216, width: 832 }, key: 'sdxl-zeta', name: 'Zeta XL' }),
  model({ format: 'lycoris', key: 'lora', name: 'Lora', type: 'lora' }),
  model({ base: 'sd-1', key: 'sd1', name: 'Dreamshaper' }),
  model({ base: 'flux', format: 'diffusers', key: 'flux-dev', name: 'FLUX Dev' }),
  model({ key: 'sdxl-alpha', name: 'Alpha XL' }),
];

const START_IMAGE: GalleryImage = {
  boardId: 'none',
  height: 750,
  imageCategory: 'user',
  imageName: 'pasted.png',
  imageUrl: '/images/pasted.png',
  queuedAt: '',
  sourceQueueItemId: '',
  starred: false,
  thumbnailUrl: '/thumbs/pasted.png',
  width: 1000,
};

const libraryEntry = (workflowId: string, name: string): WorkflowLibraryEntry => ({
  enrichment: { status: 'pending' },
  item: {
    category: 'user',
    description: `${name} description`,
    name,
    revision: 1,
    thumbnail_url: null,
    workflow_id: workflowId,
  },
  tags: [],
});

const LIBRARY: WorkflowLibraryBrowseSnapshot = {
  ...EMPTY_BROWSE,
  entries: [libraryEntry('lib-1', 'Upscale pipeline'), libraryEntry('lib-2', 'Portrait retouch')],
  pages: 1,
  status: 'loaded',
  tagCounts: [{ count: 2, tag: 'photo' }],
  total: 2,
  userTotal: 2,
};

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;

const render = () =>
  act(() =>
    root.render(
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>
          <NewProjectButton />
        </ChakraProvider>
      </QueryClientProvider>
    )
  );

const dialog = () => page.getByRole('dialog');
const start = (id: string) => page.getByRole('option', { name: `launchpad.home.intents.${id}` });
const paneHeading = () => dialog().getByRole('heading', { level: 3 });
const modelTrigger = () => dialog().getByRole('button', { name: /^projects\.newProjectDialog\.model/ });
const widthInput = () => dialog().getByRole('textbox', { name: 'projects.newProjectDialog.width' });
const heightInput = () => dialog().getByRole('textbox', { name: 'projects.newProjectDialog.height' });
const chip = (ratio: string) => dialog().getByRole('button', { name: ratio, exact: true });
const createButton = () => dialog().getByRole('button', { name: 'projects.newProjectDialog.create' });
const blankWorkflow = () => dialog().getByRole('button', { name: /projects\.newProjectDialog\.blankWorkflow/ });
const workflowCard = (workflowId: string) =>
  document.querySelector<HTMLButtonElement>(`[data-workflow-card="${workflowId}"]`)!;
const startFromRadio = (id: 'startBlank' | 'startImage') =>
  dialog().getByRole('radio', { name: `projects.newProjectDialog.${id}` });
// The segment's hidden input sits under its label; the label is what a pointer reaches.
const pickStartFrom = (id: 'startBlank' | 'startImage') =>
  userEvent.click(dialog().getByText(`projects.newProjectDialog.${id}`));
const clipboardButton = () => dialog().getByRole('button', { name: 'projects.newProjectDialog.fromClipboard' });
const sizeText = (width: number, height: number) =>
  dialog().getByText(`projects.newProjectDialog.sizeValue:height=${height},width=${width}`);

const open = async () => {
  await render();
  await userEvent.click(page.getByRole('button', { name: 'projects.newProject' }));
  await expect.element(dialog()).toBeVisible();
};

beforeEach(async () => {
  await page.viewport(1280, 900);
  vi.clearAllMocks();
  harness.canManageModels = true;
  harness.clipboard = null;
  harness.preferences = {
    enableModelDescriptions: true,
    launchpadNewProjectIntent: 'generate',
    launchpadNewProjectModelKey: null,
  };
  applyThemeToRoot(DEFAULT_THEME_ID);
  accountLifecycle.activate('new-project-test', ':user:new-project-test');
  setModelsSnapshotForTests({ error: null, models: CATALOG, status: 'loaded' });
  browse.setSnapshot!(LIBRARY);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  queryClient.clear();
  host.remove();
  takeNewProjectRequest();
  accountLifecycle.invalidate();
});

describe('NewProjectDialog', () => {
  it('opens on the remembered start with focus on it, and arrow keys move the selection and the pane', async () => {
    harness.preferences.launchpadNewProjectIntent = 'canvas';
    await open();

    await expect.element(start('canvas')).toHaveFocus();
    await expect.element(start('canvas')).toHaveAttribute('aria-selected', 'true');
    await expect.element(paneHeading()).toHaveTextContent('launchpad.home.intents.canvas');

    await userEvent.keyboard('{ArrowDown}');
    await expect.element(start('upscale')).toHaveFocus();
    await expect.element(start('upscale')).toHaveAttribute('aria-selected', 'true');
    await expect.element(start('canvas')).toHaveAttribute('aria-selected', 'false');
    await expect.element(paneHeading()).toHaveTextContent('launchpad.home.intents.upscale');
    await expect.element(dialog().getByText('projects.newProjectDialog.upscaleHint')).toBeVisible();

    await userEvent.keyboard('{Home}');
    await expect.element(start('generate')).toHaveFocus();
    await expect.element(paneHeading()).toHaveTextContent('launchpad.home.intents.generate');

    expect(await auditAccessibility(dialog().element())).toEqual([]);

    await userEvent.keyboard('{Escape}');
    await expect.element(dialog()).not.toBeInTheDocument();
    await expect.element(page.getByRole('button', { name: 'projects.newProject' })).toHaveFocus();
  });

  it('offers only generate-capable models in the shared picker, and the pick sets the default size', async () => {
    harness.preferences.launchpadNewProjectModelKey = 'sdxl-zeta';
    await open();

    await vi.waitFor(() => expect(modelTrigger().element().textContent).toContain('Zeta XL'));
    await expect.element(sizeText(832, 1216)).toBeVisible();

    await userEvent.click(modelTrigger());
    const listbox = page.getByRole('listbox', { name: /models\.scopeAvailable/ });
    await expect.element(listbox).toBeVisible();
    for (const name of ['Alpha XL', 'Dreamshaper', 'FLUX Dev', 'Zeta XL']) {
      await expect.element(listbox.getByRole('option', { name: new RegExp(name) })).toBeVisible();
    }
    expect(listbox.getByRole('option', { name: /Lora/ }).elements()).toEqual([]);

    await userEvent.click(listbox.getByRole('option', { name: /Dreamshaper/ }));
    await vi.waitFor(() => expect(modelTrigger().element().textContent).toContain('Dreamshaper'));
    await expect.element(widthInput()).toHaveValue('512');
    await expect.element(heightInput()).toHaveValue('512');
    await expect.element(chip('1:1')).toHaveAttribute('aria-pressed', 'true');
  });

  it('reshapes the size from a chip, snaps typed values to the grid, and swaps', async () => {
    harness.preferences.launchpadNewProjectModelKey = 'sdxl-alpha';
    await open();

    await userEvent.click(chip('3:2'));
    await expect.element(widthInput()).toHaveValue('1256');
    await expect.element(heightInput()).toHaveValue('840');
    await expect.element(sizeText(1256, 840)).toBeVisible();
    await expect.element(chip('3:2')).toHaveAttribute('aria-pressed', 'true');

    await userEvent.fill(widthInput(), '1003');
    await userEvent.tab();
    await expect.element(widthInput()).toHaveValue('1000');
    await expect.element(sizeText(1000, 840)).toBeVisible();
    await expect.element(chip('3:2')).toHaveAttribute('aria-pressed', 'false');

    await userEvent.click(dialog().getByRole('button', { name: 'projects.newProjectDialog.swapDimensions' }));
    await expect.element(widthInput()).toHaveValue('840');
    await expect.element(heightInput()).toHaveValue('1000');
  });

  it('keeps a chip through tabbing across the inputs and falls back from a remembered model that is gone', async () => {
    harness.preferences.launchpadNewProjectModelKey = 'uninstalled';
    await open();

    // The first selectable model by name stands in for the remembered one.
    await vi.waitFor(() => expect(modelTrigger().element().textContent).toContain('Alpha XL'));

    await userEvent.click(chip('3:2'));
    await userEvent.click(widthInput());
    await userEvent.tab();
    await userEvent.tab();
    await expect.element(chip('3:2')).toHaveAttribute('aria-pressed', 'true');
    await expect.element(widthInput()).toHaveValue('1256');

    await userEvent.click(createButton());

    expect(takeNewProjectRequest()?.generateSettings).toEqual(
      expect.objectContaining({
        aspectRatioId: '3:2',
        aspectRatioValue: 1.5,
        height: 840,
        modelKey: 'sdxl-alpha',
        width: 1256,
      })
    );
  });

  it('creates with the staged settings, remembers the start and model, and opens the editor', async () => {
    harness.preferences.launchpadNewProjectModelKey = 'sdxl-alpha';
    await open();

    await userEvent.click(start('canvas'));
    await userEvent.click(chip('16:9'));
    await userEvent.fill(dialog().getByRole('textbox', { name: 'common.name' }), '  Portraits ');
    await userEvent.click(createButton());

    expect(takeNewProjectRequest()).toEqual({
      generateSettings: expect.objectContaining({
        aspectRatioId: '16:9',
        aspectRatioIsLocked: false,
        height: 768,
        modelKey: 'sdxl-alpha',
        width: 1368,
      }),
      intent: 'canvas',
      libraryWorkflow: null,
      name: 'Portraits',
      startImage: null,
    });
    expect(harness.patchPreferences).toHaveBeenCalledWith({
      launchpadNewProjectIntent: 'canvas',
      launchpadNewProjectModelKey: 'sdxl-alpha',
    });
    expect(harness.navigate).toHaveBeenCalledWith({ search: { intent: 'canvas', new: true }, to: '/app' });
    await expect.element(dialog()).not.toBeInTheDocument();
  });

  it('creates from Enter in the name field, keeping the default name when blank', async () => {
    await open();

    await userEvent.click(start('upscale'));
    await userEvent.click(dialog().getByRole('textbox', { name: 'common.name' }));
    await userEvent.keyboard('{Enter}');

    expect(takeNewProjectRequest()).toEqual({
      generateSettings: null,
      intent: 'upscale',
      libraryWorkflow: null,
      name: null,
      startImage: null,
    });
    expect(harness.patchPreferences).toHaveBeenCalledWith({ launchpadNewProjectIntent: 'upscale' });
    expect(harness.navigate).toHaveBeenCalledWith({ search: { intent: 'upscale', new: true }, to: '/app' });
  });

  it('shows the workflow library for the Workflow start with Blank selected, and stages a picked workflow', async () => {
    await open();

    await userEvent.click(start('workflow'));
    await expect.element(blankWorkflow()).toHaveAttribute('aria-pressed', 'true');
    expect(browse.ensureWorkflowLibraryBrowseLoaded).toHaveBeenCalledTimes(1);
    await expect
      .element(dialog().getByRole('tab', { name: 'workflowLibrary.yours' }))
      .toHaveAttribute('aria-selected', 'true');
    await vi.waitFor(() => expect(workflowCard('lib-1')).not.toBeNull());
    expect(document.querySelector('[data-workflow-card-menu]')).toBeNull();
    // The library card's "not run yet" placeholder is a known low-contrast label; nothing else may fail.
    const violations = await auditAccessibility(dialog().element());
    expect(violations.map((violation) => violation.id)).toEqual(['color-contrast']);
    expect(contrastOffenderTexts(violations)).toEqual(['workflowLibrary.notRunYet', 'workflowLibrary.notRunYet']);

    await userEvent.click(workflowCard('lib-1'));
    await expect.element(blankWorkflow()).toHaveAttribute('aria-pressed', 'false');
    expect(workflowCard('lib-1').getAttribute('aria-pressed')).toBe('true');

    await userEvent.click(blankWorkflow());
    await expect.element(blankWorkflow()).toHaveAttribute('aria-pressed', 'true');
    expect(workflowCard('lib-1').getAttribute('aria-pressed')).toBe('false');

    await userEvent.click(workflowCard('lib-2'));
    await userEvent.click(createButton());

    expect(takeNewProjectRequest()).toEqual({
      generateSettings: null,
      intent: 'workflow',
      libraryWorkflow: { name: 'Portrait retouch', workflowId: 'lib-2' },
      name: null,
      startImage: null,
    });
    expect(harness.navigate).toHaveBeenCalledWith({ search: { intent: 'workflow', new: true }, to: '/app' });
  });

  it('creates straight from Enter on a focused workflow card', async () => {
    await open();

    await userEvent.click(start('workflow'));
    await vi.waitFor(() => expect(workflowCard('lib-1')).not.toBeNull());
    await userEvent.click(workflowCard('lib-1'));
    await userEvent.keyboard('{Enter}');

    expect(takeNewProjectRequest()?.libraryWorkflow).toEqual({ name: 'Upscale pipeline', workflowId: 'lib-1' });
    await expect.element(dialog()).not.toBeInTheDocument();
  });

  it('starts a Canvas project from a pasted image, sizing the canvas to it', async () => {
    harness.preferences.launchpadNewProjectModelKey = 'sdxl-alpha';
    harness.upload.mockResolvedValue(START_IMAGE);
    await open();

    await userEvent.click(start('canvas'));
    await expect.element(startFromRadio('startBlank')).toBeChecked();
    expect(clipboardButton().elements()).toEqual([]);

    await pickStartFrom('startImage');
    await expect.element(dialog().getByRole('button', { name: 'widgets.gallery.picker.chooseImage' })).toBeVisible();
    // Image chosen as the start but none picked yet: Create waits and says why.
    await expect.element(createButton()).toBeDisabled();
    await expect.element(dialog().getByText('projects.newProjectDialog.chooseImageOrBlank')).toBeVisible();

    // Nothing on the clipboard is an ordinary outcome, said inline rather than toasted.
    await userEvent.click(clipboardButton());
    await expect.element(dialog().getByText('projects.newProjectDialog.clipboardEmpty')).toBeVisible();
    expect(harness.upload).not.toHaveBeenCalled();

    harness.clipboard = new Blob(['png'], { type: 'image/png' });
    await userEvent.click(clipboardButton());
    await vi.waitFor(() => expect(harness.upload).toHaveBeenCalledTimes(1));
    const [file, boardId] = harness.upload.mock.calls[0] as [File, string];
    expect(file.name).toBe('clipboard.png');
    expect(file.type).toBe('image/png');
    expect(boardId).toBe('none');

    // The image sizes the canvas: 1000×750 on SDXL's 8px grid is 1000×752, and the size controls rest.
    await expect.element(widthInput()).toHaveValue('1000');
    await expect.element(heightInput()).toHaveValue('752');
    await expect.element(widthInput()).toBeDisabled();
    await expect.element(chip('3:2')).toBeDisabled();
    await expect.element(sizeText(1000, 752)).toBeVisible();
    await expect.element(dialog().getByText('projects.newProjectDialog.imageSizeHint')).toBeVisible();
    await expect.element(dialog().getByText('projects.newProjectDialog.clipboardEmpty')).not.toBeInTheDocument();
    await expect.element(createButton()).toBeEnabled();
    expect(await auditAccessibility(dialog().element())).toEqual([]);

    // Blank forgets the image and frees the size again.
    await pickStartFrom('startBlank');
    await expect.element(widthInput()).toBeEnabled();
    await expect.element(widthInput()).toHaveValue('1024');
    await pickStartFrom('startImage');
    await expect.element(widthInput()).toHaveValue('1024');

    await userEvent.click(clipboardButton());
    await vi.waitFor(() => expect(harness.upload).toHaveBeenCalledTimes(2));
    await expect.element(widthInput()).toHaveValue('1000');
    await userEvent.click(createButton());

    expect(takeNewProjectRequest()).toEqual({
      generateSettings: expect.objectContaining({ aspectRatioId: '4:3', height: 752, width: 1000 }),
      intent: 'canvas',
      libraryWorkflow: null,
      name: null,
      startImage: expect.objectContaining({ height: 750, imageName: 'pasted.png', width: 1000 }),
    });
  });

  it('runs the import flow from the rail and closes', async () => {
    await open();

    await userEvent.click(dialog().getByRole('button', { name: 'projects.newProjectDialog.importFile' }));

    expect(harness.importFile).toHaveBeenCalledTimes(1);
    await expect.element(dialog()).not.toBeInTheDocument();
  });

  it('blocks Create without a compatible model and points admins at the model manager', async () => {
    setModelsSnapshotForTests({ models: CATALOG.filter((entry) => entry.type === 'lora'), status: 'loaded' });
    await open();

    await expect.element(dialog().getByText('projects.newProjectDialog.noModelsTitle').first()).toBeVisible();
    const managerLink = dialog().getByRole('link', { name: 'projects.newProjectDialog.openModelManager' });
    await expect.element(managerLink).toHaveAttribute('href', '/models');
    await expect.element(createButton()).toBeDisabled();
    expect(await auditAccessibility(dialog().element())).toEqual([]);

    await userEvent.click(start('workflow'));
    await expect.element(createButton()).toBeEnabled();
  });

  it('hides the model manager link from users who cannot manage models', async () => {
    harness.canManageModels = false;
    setModelsSnapshotForTests({ models: [], status: 'loaded' });
    await open();

    await expect.element(dialog().getByText('projects.newProjectDialog.noModelsTitle').first()).toBeVisible();
    expect(dialog().getByRole('link', { name: 'projects.newProjectDialog.openModelManager' }).elements()).toEqual([]);
  });
});
