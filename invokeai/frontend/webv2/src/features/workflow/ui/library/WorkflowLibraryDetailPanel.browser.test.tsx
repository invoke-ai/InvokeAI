import type { StarterModel } from '@features/models';
import type { WorkflowModelRequirement } from '@features/workflow/core/modelRequirements';
import type { WorkflowLibraryEntry, WorkflowLibraryEntryEnrichment } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowGraphPreviewPort, WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph, serializeWorkflowJson } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowLibraryDetailPanel } from './WorkflowLibraryDetailPanel';

/**
 * The panel is the only surface that turns "what does this workflow need" into
 * an action, so the model stores it resolves against are replaced by mutable
 * fixtures: installed models, the starter catalog, and the set of sources
 * already installing. `getStarterModelInstallSources` stays real — the install
 * action's contract is that it hands the *catalog's* sources to `installMany`,
 * deduped across requirements.
 */
const models = vi.hoisted(() => ({
  activeInstallSources: { current: new Set<string>() },
  ensureModelsLoaded: vi.fn(() => Promise.resolve()),
  ensureStartersLoaded: vi.fn(),
  installedModels: { current: [] as unknown[] },
  installMany: vi.fn((requests: readonly unknown[]) => Promise.resolve(requests.length)),
  starterModels: { current: [] as unknown[] },
}));

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: models.ensureModelsLoaded,
  ensureStartersLoaded: models.ensureStartersLoaded,
  useActiveInstallSources: () => models.activeInstallSources.current,
  useInstallActions: () => ({
    install: models.installMany,
    installMany: models.installMany,
    pendingSources: new Set(),
  }),
  useModelsSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({
      coverImageVersions: {},
      error: null,
      missingModelKeys: new Set(),
      models: models.installedModels.current,
      modelsByKey: new Map(),
      modelsDir: null,
      status: 'loaded',
    }),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({
      error: null,
      response: { starter_bundles: {}, starter_models: models.starterModels.current },
      status: 'loaded',
    }),
}));

const queries = vi.hoisted(() => ({
  createLibraryWorkflow: vi.fn((_workflow: Record<string, unknown>, _signal?: AbortSignal) =>
    Promise.resolve('wf-copy')
  ),
  deleteLibraryWorkflow: vi.fn((_workflowId: string, _signal?: AbortSignal) => Promise.resolve()),
  getLibraryWorkflowCached: vi.fn((_workflowId: string, _signal?: AbortSignal) =>
    Promise.resolve({} as Record<string, unknown>)
  ),
  invalidateWorkflowLibraryCache: vi.fn(),
}));

vi.mock('@features/workflow/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ...queries,
}));

const downloadText = vi.hoisted(() => vi.fn((_contents: string, _fileName: string, _mimeType: string) => {}));

vi.mock('@platform/browser/downloadBlob', () => ({ downloadText }));

const TRANSLATIONS: Record<string, string> = {
  'common.unknownError': 'Something went wrong',
  'workflowLibrary.delete': 'Delete',
  'workflowLibrary.deleteConfirmBody': 'Delete "{{name}}" from the workflow library? This cannot be undone.',
  'workflowLibrary.deleteConfirmTitle': 'Delete workflow',
  'workflowLibrary.deleteFailed': 'Failed to delete workflow',
  'workflowLibrary.downloadJson': 'Download JSON',
  'workflowLibrary.duplicate': 'Duplicate',
  'workflowLibrary.duplicateFailed': 'Failed to duplicate workflow',
  'workflowLibrary.duplicateName': '{{name}} copy',
  'workflowLibrary.forkIntoProject': 'Fork into new project',
  'workflowLibrary.installModels_one': 'Install 1 model',
  'workflowLibrary.installModels_other': 'Install {{count}} models',
  'workflowLibrary.installQueued': 'Model installs queued',
  'workflowLibrary.lastRun': 'Your last run · {{when}}',
  'workflowLibrary.loadFailed': 'Failed to load workflow',
  'workflowLibrary.moreActions': 'More actions',
  'workflowLibrary.notRunYet': 'Not run yet',
  'workflowLibrary.open': 'Open',
  'workflowLibrary.previewGraph': 'Preview graph',
  'workflowLibrary.requirementInstallable': 'Not installed',
  'workflowLibrary.requirementInstalled': 'Installed',
  'workflowLibrary.requirementInstalling': 'Installing…',
  'workflowLibrary.requirementMissing': 'Not available to install',
  'workflowLibrary.requires': 'Requires',
  'workflowLibrary.sampleOutput': 'Sample output',
  'workflowLibrary.untitled': 'Untitled Workflow',
};

const interpolate = (template: string, options?: Record<string, unknown>): string =>
  options ? template.replaceAll(/\{\{(\w+)\}\}/g, (_match, key: string) => String(options[key] ?? '')) : template;

const translate = (key: string, options?: Record<string, unknown>): string => {
  const count = options?.count;
  const plural = typeof count === 'number' ? TRANSLATIONS[`${key}_${count === 1 ? 'one' : 'other'}`] : undefined;

  return interpolate(plural ?? TRANSLATIONS[key] ?? key, options);
};

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: translate }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// #region Fixtures

const WAN_VAE_DEPENDENCY = {
  base: 'wan',
  description: 'Wan VAE',
  is_installed: false,
  name: 'Wan VAE',
  source: 'https://models.test/wan-vae',
  type: 'vae',
} as const;

/** Two starters sharing one dependency — the install action must send it once. */
const WAN_MAIN_STARTER: StarterModel = {
  base: 'wan',
  dependencies: [{ ...WAN_VAE_DEPENDENCY }],
  description: 'Wan 2.2 I2V A14B',
  is_installed: false,
  name: 'Wan 2.2 I2V A14B',
  source: 'https://models.test/wan-i2v',
  type: 'main',
};

const WAN_LORA_STARTER: StarterModel = {
  base: 'wan',
  dependencies: [{ ...WAN_VAE_DEPENDENCY }],
  description: 'Lightning LoRA',
  is_installed: false,
  name: 'Lightning LoRA',
  source: 'https://models.test/lightning-lora',
  type: 'lora',
};

const INSTALLED_WAN_VAE = {
  base: 'wan',
  hash: 'hash-wan-vae',
  key: 'installed-wan-vae',
  name: 'Wan VAE',
  type: 'vae',
};

const INSTALLED_SDXL_MAIN = {
  base: 'sdxl',
  hash: 'hash-sdxl',
  key: 'installed-sdxl',
  name: 'SDXL 1.0',
  type: 'main',
};

const slot = (base: string, modelType: string, label: string): WorkflowModelRequirement => ({
  base,
  kind: 'slot',
  label,
  modelType,
});

const SDXL_REQUIREMENTS: readonly WorkflowModelRequirement[] = [slot('sdxl', 'main', 'SDXL 1.0')];
const WAN_REQUIREMENTS: readonly WorkflowModelRequirement[] = [
  slot('wan', 'main', 'Wan 2.2 I2V A14B'),
  slot('wan', 'lora', 'Lightning LoRA'),
  slot('wan', 'vae', 'Wan VAE'),
];

const readyEnrichment = (requirements: readonly WorkflowModelRequirement[]): WorkflowLibraryEntryEnrichment => ({
  document: createProjectGraph('library-fixture'),
  nodeCount: requirements.length,
  requirements: { primaryBase: null, requirements },
  status: 'ready',
});

const entry = (
  overrides: Partial<WorkflowLibraryEntry['item']> & { workflow_id: string; name: string },
  enrichment: WorkflowLibraryEntryEnrichment,
  tags: readonly string[] = []
): WorkflowLibraryEntry => ({
  enrichment,
  item: {
    category: 'user',
    description: `${overrides.name} description`,
    thumbnail_url: null,
    ...overrides,
  },
  tags,
});

const TEXT_TO_IMAGE = entry(
  {
    description: 'The baseline SDXL graph.',
    last_run_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    name: 'Text to image',
    thumbnail_url: 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==',
    workflow_id: 'wf-text-to-image',
  },
  readyEnrichment(SDXL_REQUIREMENTS),
  ['SDXL', 'text to image']
);

const IMAGE_TO_VIDEO = entry(
  {
    description: 'Animates a still image.',
    name: 'Image to video',
    thumbnail_url: 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==',
    workflow_id: 'wf-image-to-video',
  },
  readyEnrichment(WAN_REQUIREMENTS),
  ['Wan 2.2']
);

const DEFAULT_CATEGORY = entry(
  { category: 'default', name: 'Bundled default', workflow_id: 'wf-default' },
  readyEnrichment(SDXL_REQUIREMENTS)
);

const RAW_WORKFLOW: Record<string, unknown> = {
  ...serializeWorkflowJson({ ...createProjectGraph('raw-fixture'), name: 'Image to video' }),
  id: 'wf-image-to-video',
  meta: { category: 'default', version: '3.0.0' },
};

// #endregion

const NOTIFICATIONS = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
const ADAPTER = { notifications: NOTIFICATIONS } as unknown as WorkflowUiAdapter;
const OPEN_DOCUMENT_IN_NEW_PROJECT = vi.fn();
const GRAPH_PREVIEW = {
  openDocumentInNewProject: OPEN_DOCUMENT_IN_NEW_PROJECT,
} as unknown as WorkflowGraphPreviewPort;

describe('WorkflowLibraryDetailPanel', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onClose: () => void;
  let onDeleted: () => void;
  let onDuplicated: (workflowId: string) => void;
  let onOpen: (item: WorkflowLibraryEntry['item']) => void;
  let onPreview: (selected: WorkflowLibraryEntry) => void;

  /** See `WorkflowLibraryDialog.browser.test.tsx`: keeps Chakra's observer-driven commits inside the act scope. */
  const settleFrame = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

  const renderPanel = async (selected: WorkflowLibraryEntry | null) => {
    await act(async () => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={ADAPTER}>
              <WorkflowGraphPreviewProvider adapter={GRAPH_PREVIEW}>
                <WorkflowLibraryDetailPanel
                  entry={selected}
                  onClose={onClose}
                  onDeleted={onDeleted}
                  onDuplicated={onDuplicated}
                  onOpen={onOpen}
                  onPreview={onPreview}
                />
              </WorkflowGraphPreviewProvider>
            </WorkflowUiProvider>
          </ChakraProvider>
        </StrictMode>
      );
      await settleFrame();
    });
  };

  const panel = () => document.querySelector<HTMLElement>('[data-workflow-detail]');
  const buttonWithText = (text: string) =>
    [...document.querySelectorAll('button')].find((candidate) => (candidate.textContent ?? '').trim() === text);
  const requirementRows = () => [...document.querySelectorAll<HTMLElement>('[data-requirement-status]')];
  const requirementStatuses = () => requirementRows().map((row) => row.dataset.requirementStatus);

  const clickButton = async (text: string) => {
    const button = buttonWithText(text);
    expect(button, `no button labelled "${text}"`).not.toBeUndefined();

    await act(async () => {
      button?.click();
      await settleFrame();
    });
  };

  const openMenu = async () => {
    const trigger = document.querySelector<HTMLElement>('[aria-label="More actions"]');
    expect(trigger).not.toBeNull();

    await act(async () => {
      trigger?.click();
      await settleFrame();
    });
  };

  const menuItem = (value: string) => document.querySelector<HTMLElement>(`[data-menu-item="${value}"]`);

  const clickMenuItem = async (value: string) => {
    await openMenu();

    const item = menuItem(value);
    expect(item, `no menu item "${value}"`).not.toBeNull();

    await act(async () => {
      item?.click();
      await settleFrame();
    });
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    onClose = vi.fn();
    onDeleted = vi.fn();
    onDuplicated = vi.fn((_workflowId: string) => {});
    onOpen = vi.fn();
    onPreview = vi.fn();

    models.activeInstallSources.current = new Set();
    models.installedModels.current = [INSTALLED_SDXL_MAIN, INSTALLED_WAN_VAE];
    models.starterModels.current = [WAN_MAIN_STARTER, WAN_LORA_STARTER];
    models.ensureModelsLoaded.mockClear();
    models.ensureStartersLoaded.mockClear();
    models.installMany.mockClear();

    queries.createLibraryWorkflow.mockClear();
    queries.createLibraryWorkflow.mockResolvedValue('wf-copy');
    queries.deleteLibraryWorkflow.mockClear();
    queries.deleteLibraryWorkflow.mockResolvedValue(undefined);
    queries.getLibraryWorkflowCached.mockClear();
    queries.getLibraryWorkflowCached.mockResolvedValue(RAW_WORKFLOW);
    queries.invalidateWorkflowLibraryCache.mockClear();

    downloadText.mockClear();
    OPEN_DOCUMENT_IN_NEW_PROJECT.mockClear();
    NOTIFICATIONS.error.mockClear();
    NOTIFICATIONS.info.mockClear();
    NOTIFICATIONS.success.mockClear();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('renders the workflow facts and loads the model data it resolves against', async () => {
    await renderPanel(TEXT_TO_IMAGE);

    const text = panel()?.textContent ?? '';

    expect(text).toContain('Text to image');
    expect(text).toContain('The baseline SDXL graph.');
    expect(text).toContain('SDXL');
    expect(text).toContain('text to image');
    expect(text).toContain('Requires');
    expect(models.ensureModelsLoaded).toHaveBeenCalled();
    expect(models.ensureStartersLoaded).toHaveBeenCalled();
  });

  it('offers Open as the primary action when every requirement is installed', async () => {
    await renderPanel(TEXT_TO_IMAGE);

    expect(requirementStatuses()).toEqual(['installed']);
    expect(buttonWithText('Open')).not.toBeUndefined();
    expect(buttonWithText('Install 1 model')).toBeUndefined();

    await clickButton('Open');

    expect(onOpen).toHaveBeenCalledWith(TEXT_TO_IMAGE.item);
  });

  it('opens the workflow from the keyboard, the path the double-click-only cards do not offer', async () => {
    await renderPanel(TEXT_TO_IMAGE);

    const open = buttonWithText('Open');
    expect(open?.tagName).toBe('BUTTON');

    await act(async () => {
      open?.focus();
      expect(document.activeElement).toBe(open);
      open?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' }));
      // A real Enter on a focused button dispatches a click; jsdom-style
      // synthesis does not, so the click is what the browser would send next.
      open?.click();
      await settleFrame();
    });

    expect(onOpen).toHaveBeenCalledWith(TEXT_TO_IMAGE.item);
  });

  it('swaps the primary action for an install when starter models can fill the gaps', async () => {
    await renderPanel(IMAGE_TO_VIDEO);

    expect(requirementStatuses()).toEqual(['installable', 'installable', 'installed']);
    // The rows say which state they are in, not just how they are styled.
    expect(document.querySelectorAll('[aria-label="Not installed"]')).toHaveLength(2);
    expect(document.querySelectorAll('[aria-label="Installed"]')).toHaveLength(1);
    expect(buttonWithText('Install 2 models')).not.toBeUndefined();
    expect(buttonWithText('Open')).toBeUndefined();
  });

  it('queues one install per source, deduped across requirements that share a dependency', async () => {
    await renderPanel(IMAGE_TO_VIDEO);

    await clickButton('Install 2 models');

    expect(models.installMany).toHaveBeenCalledTimes(1);

    const requests = models.installMany.mock.calls[0]?.[0] as { source: string }[];

    expect(requests.map((request) => request.source)).toEqual([
      'https://models.test/wan-vae',
      'https://models.test/wan-i2v',
      'https://models.test/lightning-lora',
    ]);
    expect(NOTIFICATIONS.success).toHaveBeenCalledWith('Model installs queued');
  });

  it('shows the in-flight rows as installing and drops the install action', async () => {
    models.activeInstallSources.current = new Set(['https://models.test/wan-i2v', 'https://models.test/wan-vae']);

    await renderPanel(IMAGE_TO_VIDEO);

    expect(requirementStatuses()).toEqual(['installing', 'installing', 'installed']);
    expect(document.querySelectorAll('[aria-label="Installing…"]')).toHaveLength(2);
    expect(buttonWithText('Install 2 models')).toBeUndefined();
    expect(buttonWithText('Open')).not.toBeUndefined();
  });

  it('marks a requirement with no installed model and no starter as unresolvable', async () => {
    models.starterModels.current = [];

    await renderPanel(IMAGE_TO_VIDEO);

    expect(requirementStatuses()).toEqual(['unresolvable', 'unresolvable', 'installed']);
    // Nothing to install, so the panel keeps its Open action rather than
    // offering an install that cannot run.
    expect(buttonWithText('Open')).not.toBeUndefined();
  });

  it('captions the thumbnail with the last run, the sample output, or nothing', async () => {
    await renderPanel(TEXT_TO_IMAGE);
    expect(panel()?.textContent).toContain('Your last run · 2 days ago');

    await renderPanel(IMAGE_TO_VIDEO);
    expect(panel()?.textContent).toContain('Sample output');
    expect(panel()?.textContent).not.toContain('Your last run');

    await renderPanel(DEFAULT_CATEGORY);
    expect(panel()?.textContent).not.toContain('Sample output');
    expect(panel()?.textContent).toContain('Not run yet');
  });

  it('updates in place when the selection changes', async () => {
    await renderPanel(TEXT_TO_IMAGE);

    const before = panel();
    expect(before?.dataset.workflowDetail).toBe('wf-text-to-image');

    await renderPanel(IMAGE_TO_VIDEO);

    // Same element, new content: a keyed remount would flash the whole rail.
    expect(panel()).toBe(before);
    expect(panel()?.dataset.workflowDetail).toBe('wf-image-to-video');
  });

  it('renders placeholder rows while enrichment is pending and a quiet line when it failed', async () => {
    await renderPanel(entry({ name: 'Pending', workflow_id: 'wf-pending' }, { status: 'pending' }));

    expect(document.querySelectorAll('[data-requirement-placeholder]').length).toBeGreaterThan(0);
    expect(requirementRows()).toHaveLength(0);

    await renderPanel(
      entry({ name: 'Broken', workflow_id: 'wf-broken' }, { message: 'Failed to read this workflow.', status: 'error' })
    );

    expect(panel()?.textContent).toContain('Failed to read this workflow.');
    expect(document.querySelectorAll('[data-requirement-placeholder]')).toHaveLength(0);
  });

  it('hides Delete for bundled defaults but still offers Duplicate', async () => {
    await renderPanel(DEFAULT_CATEGORY);
    await openMenu();

    expect(menuItem('delete')).toBeNull();
    expect(menuItem('duplicate')).not.toBeNull();

    await renderPanel(TEXT_TO_IMAGE);
    await openMenu();

    expect(menuItem('delete')).not.toBeNull();
  });

  it('duplicates the library record without its id, renamed, and owned by the user', async () => {
    await renderPanel(DEFAULT_CATEGORY);

    await clickMenuItem('duplicate');

    expect(queries.getLibraryWorkflowCached).toHaveBeenCalledWith('wf-default', expect.anything());
    expect(queries.createLibraryWorkflow).toHaveBeenCalledTimes(1);

    const created = queries.createLibraryWorkflow.mock.calls[0]?.[0] as Record<string, unknown>;

    expect(created).not.toHaveProperty('id');
    expect(created.name).toBe('Bundled default copy');
    expect(created.meta).toStrictEqual({ category: 'user', version: '3.0.0' });
    // The original record is untouched — only a copy is written.
    expect(RAW_WORKFLOW.meta).toStrictEqual({ category: 'default', version: '3.0.0' });
    expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledTimes(1);
    expect(onDuplicated).toHaveBeenCalledWith('wf-copy');
  });

  it('forks the cached workflow into a fresh project', async () => {
    await renderPanel(IMAGE_TO_VIDEO);

    await clickMenuItem('fork-into-project');

    expect(OPEN_DOCUMENT_IN_NEW_PROJECT).toHaveBeenCalledTimes(1);

    const [document_, label] = OPEN_DOCUMENT_IN_NEW_PROJECT.mock.calls[0] as [{ name: string }, string];

    expect(document_.name).toBe('Image to video');
    expect(label).toBe('Image to video');
    // The fork lands the user in a new project, so the library gets out of the way.
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(queries.createLibraryWorkflow).not.toHaveBeenCalled();
  });

  it('downloads the raw workflow JSON under a slugged file name', async () => {
    await renderPanel(IMAGE_TO_VIDEO);

    await clickMenuItem('download-json');

    expect(downloadText).toHaveBeenCalledTimes(1);

    const [contents, fileName, mimeType] = downloadText.mock.calls[0] as [string, string, string];

    expect(JSON.parse(contents)).toStrictEqual(RAW_WORKFLOW);
    expect(fileName).toBe('image-to-video.json');
    expect(mimeType).toBe('application/json');
  });

  it('deletes only after the confirmation is accepted', async () => {
    await renderPanel(TEXT_TO_IMAGE);

    await clickMenuItem('delete');

    const confirm = document.querySelector<HTMLElement>('[role="alertdialog"]');
    expect(confirm?.textContent).toContain('Delete workflow');
    expect(queries.deleteLibraryWorkflow).not.toHaveBeenCalled();

    const confirmButton = [...(confirm?.querySelectorAll('button') ?? [])].find(
      (candidate) => (candidate.textContent ?? '').trim() === 'Delete'
    );
    expect(confirmButton).not.toBeUndefined();

    await act(async () => {
      confirmButton?.click();
      await settleFrame();
    });

    expect(queries.deleteLibraryWorkflow).toHaveBeenCalledWith('wf-text-to-image', expect.anything());
    expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledTimes(1);
    expect(onDeleted).toHaveBeenCalledTimes(1);
  });

  it('reports a failed delete without claiming the workflow is gone', async () => {
    queries.deleteLibraryWorkflow.mockRejectedValue(new Error('network down'));

    await renderPanel(TEXT_TO_IMAGE);
    await clickMenuItem('delete');

    const confirm = document.querySelector<HTMLElement>('[role="alertdialog"]');
    const confirmButton = [...(confirm?.querySelectorAll('button') ?? [])].find(
      (candidate) => (candidate.textContent ?? '').trim() === 'Delete'
    );

    await act(async () => {
      confirmButton?.click();
      await settleFrame();
    });

    expect(NOTIFICATIONS.error).toHaveBeenCalledWith('Failed to delete workflow', expect.any(String));
    expect(onDeleted).not.toHaveBeenCalled();
    expect(queries.invalidateWorkflowLibraryCache).not.toHaveBeenCalled();
  });

  it('hands the selected entry to the preview action', async () => {
    await renderPanel(IMAGE_TO_VIDEO);

    await clickButton('Preview graph');

    expect(onPreview).toHaveBeenCalledWith(IMAGE_TO_VIDEO);
  });

  it('renders nothing actionable without a selection', async () => {
    await renderPanel(null);

    expect(buttonWithText('Open')).toBeUndefined();
    expect(document.querySelector('[aria-label="More actions"]')).toBeNull();
  });
});
