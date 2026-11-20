import type { GalleryHost } from '@features/gallery/picker';
import type { StarterModel } from '@features/models';
import type { WorkflowModelRequirement } from '@features/workflow/core/modelRequirements';
import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryEntry, WorkflowLibraryEntryEnrichment } from '@features/workflow/data/libraryBrowseStore';
import type { LibraryWorkflowReadOptions } from '@features/workflow/data/libraryCache';
import type { WorkflowGraphPreviewPort, WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { GalleryHostProvider } from '@features/gallery/picker';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph, serializeWorkflowJson } from '@features/workflow/utility';
import { resolvedValues } from '@platform/core/resolvedValues.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act, StrictMode, type ComponentProps, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { loadGraphPreview } from './libraryPreviewSource';
import { useWorkflowLibraryDuplicateGuard } from './useWorkflowLibraryEntryActions';
import { WorkflowLibraryDetailPanel } from './WorkflowLibraryDetailPanel';

/** As the template panes mount it; for this single-user owner only the account's own templates are writable. */
const PanesRail = (props: Omit<ComponentProps<typeof WorkflowLibraryDetailPanel>, 'duplicateGuard' | 'isReadOnly'>) => (
  <WorkflowLibraryDetailPanel
    {...props}
    duplicateGuard={useWorkflowLibraryDuplicateGuard()}
    isReadOnly={props.entry?.item.category !== 'user'}
  />
);

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
  deleteLibraryWorkflowThumbnail: vi.fn((_workflowId: string, _signal?: AbortSignal) => Promise.resolve()),
  getLibraryWorkflowCached: vi.fn((_workflowId: string, _options?: LibraryWorkflowReadOptions) =>
    Promise.resolve({} as Record<string, unknown>)
  ),
  getLibraryWorkflowRecord: vi.fn((_workflowId: string, _signal?: AbortSignal) =>
    Promise.resolve({} as Record<string, unknown>)
  ),
  getLibraryWorkflowRecordCached: vi.fn((_workflowId: string, _options?: LibraryWorkflowReadOptions) =>
    Promise.resolve({} as Record<string, unknown>)
  ),
  invalidateWorkflowLibraryCache: vi.fn(),
  setLibraryWorkflowThumbnail: vi.fn((_workflowId: string, _image: Blob, _signal?: AbortSignal) => Promise.resolve()),
  setLibraryWorkflowPublic: vi.fn((_workflowId: string, isPublic: boolean, _signal?: AbortSignal) =>
    Promise.resolve({ is_public: isPublic })
  ),
  updateLibraryWorkflow: vi.fn(
    (_workflowId: string, _workflow: Record<string, unknown>, _options?: { expectedRevision?: number }) =>
      Promise.resolve({} as Record<string, unknown>)
  ),
}));

vi.mock('@features/workflow/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ...queries,
}));

const GALLERY_HOST: GalleryHost = {
  galleryValues: resolvedValues(),
  notifications: { add: vi.fn(), reportError: vi.fn() },
  projectName: '',
};

const pickedGalleryItem = vi.hoisted(() => ({
  current: { kind: 'image', name: 'picked.gif', thumbnailUrl: '' } as Record<string, unknown>,
}));

const identity = vi.hoisted(() => ({
  session: { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } } as {
    multiuserEnabled: boolean;
    phase: 'ready';
    user: { user_id: string } | null;
  },
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuthSession: () => identity.session,
}));

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationTemplatesSnapshot: () => ({ error: null, status: 'loaded', templates: {} }),
}));

const SNAPSHOT_IMAGE = vi.hoisted(() => new Blob(['snapshot'], { type: 'image/png' }));
const snapshotCapture = vi.hoisted(() => ({ settles: true }));
vi.mock('@features/workflow/ui/graph-preview/GraphPreviewSnapshot', () => ({
  GraphPreviewSnapshot: ({ onCapture }: { onCapture: (image: Blob) => void }) => {
    if (snapshotCapture.settles) {
      queueMicrotask(() => onCapture(SNAPSHOT_IMAGE));
    }
    return null;
  },
}));

vi.mock('@features/gallery/ui/picker/GalleryPickerPopover', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  GalleryPickerPopover: ({ children, onPick }: { children: ReactNode; onPick: (item: unknown) => void }) => (
    <>
      {children}
      <button aria-label="Pick" data-gallery-pick type="button" onClick={() => onPick(pickedGalleryItem.current)} />
    </>
  ),
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
  'workflowLibrary.rename': 'Rename',
  'workflowLibrary.renameFailed': 'Rename failed',
  'workflowLibrary.renameTemplateTitle': 'Rename template',
  'workflowLibrary.renameWithEllipsis': 'Rename…',
  'workflowLibrary.renamed': 'Template renamed',
  'workflowLibrary.templateName': 'Template name',
  'workflowLibrary.thumbnail': 'Thumbnail',
  'workflowLibrary.thumbnailChoose': 'Choose thumbnail',
  'workflowLibrary.thumbnailRemove': 'Remove thumbnail',
  'workflowLibrary.thumbnailRemoveFailed': 'Failed to remove thumbnail',
  'workflowLibrary.thumbnailRemoved': 'Thumbnail removed',
  'workflowLibrary.thumbnailSnapshot': 'Snapshot the graph',
  'workflowLibrary.thumbnailUpdateFailed': 'Failed to update thumbnail',
  'workflowLibrary.thumbnailUpdated': 'Thumbnail updated',
  'workflowLibrary.thumbnailUpdating': 'Updating thumbnail…',
  'workflowLibrary.duplicateName': '{{name}} copy',
  'workflowLibrary.duplicated': 'Saved a copy under Yours',
  'workflowLibrary.findModel': 'Find {{name}} in the model manager',
  'workflowLibrary.forkIntoProject': 'Fork into new project',
  'workflowLibrary.installModels_one': 'Install 1 model',
  'workflowLibrary.installModels_other': 'Install {{count}} models',
  'workflowLibrary.installQueued': 'Model installs queued',
  'workflowLibrary.lastRun': 'Your last run · {{when}}',
  'workflowLibrary.loadFailed': 'Failed to load workflow',
  'workflowLibrary.moreActions': 'More actions',
  'workflowLibrary.notRunYet': 'Not run yet',
  'workflowLibrary.open': 'Open',
  'workflowLibrary.openWithEllipsis': 'Open…',
  'workflowLibrary.preview.action': 'Preview',
  'workflowLibrary.requirementInstallable': 'Not installed',
  'workflowLibrary.requirementInstalled': 'Installed',
  'workflowLibrary.requirementInstalling': 'Installing…',
  'workflowLibrary.requirementMissing': 'Not available to install',
  'workflowLibrary.requires': 'Requires',
  'workflowLibrary.sampleOutput': 'Sample output',
  'workflowLibrary.untitled': 'Untitled Workflow',
  'workflowLibrary.private': 'Private',
  'workflowLibrary.public': 'Public',
  'workflowLibrary.publicVisibilityHelp': 'Visible to users of this InvokeAI instance.',
  'workflowLibrary.publicWorkflow': 'Public workflow',
  'workflowLibrary.visibilityUpdateFailed': 'Failed to update workflow visibility',
  'workflowLibrary.visibilityUnknown': 'Visibility unknown',
  'workflowLibrary.retry': 'Retry',
  'workflowLibrary.modelsToInstall_one': '{{count}} model to install',
  'workflowLibrary.modelsToInstall_other': '{{count}} models to install',
  'workflowLibrary.nodeCount_one': '{{count}} node',
  'workflowLibrary.nodeCount_other': '{{count}} nodes',
  'workflowLibrary.openedWhen': 'Opened {{when}}',
  'workflowLibrary.readyToRun': 'Ready to run',
};

const interpolate = (template: string, options?: Record<string, unknown>): string =>
  options ? template.replaceAll(/\{\{(\w+)\}\}/g, (_match, key: string) => String(options[key] ?? '')) : template;

const translate = (key: string, options?: Record<string, unknown>): string => {
  const count = options?.count;
  const plural = typeof count === 'number' ? TRANSLATIONS[`${key}_${count === 1 ? 'one' : 'other'}`] : undefined;

  return interpolate(plural ?? TRANSLATIONS[key] ?? key, options);
};

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ i18n: { language: 'en', resolvedLanguage: 'en' }, t: translate }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const NOOP = () => {};

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
    revision: 1,
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

const privateWorkflowEntry = (base: WorkflowLibraryEntry, workflowId: string): WorkflowLibraryEntry => ({
  ...base,
  item: { ...base.item, is_public: false, user_id: 'user-a', workflow_id: workflowId },
});

const DEFAULT_CATEGORY = entry(
  { category: 'default', name: 'Bundled default', workflow_id: 'wf-default' },
  readyEnrichment(SDXL_REQUIREMENTS)
);

const RAW_WORKFLOW: Record<string, unknown> = {
  ...serializeWorkflowJson({ ...createProjectGraph('raw-fixture'), name: 'Image to video' }),
  id: 'wf-image-to-video',
  meta: { category: 'default', version: '3.0.0' },
};

/** The library record behind IMAGE_TO_VIDEO, at the revision a fork must carry along. */
const RAW_RECORD = {
  created_at: '2026-10-20T00:00:00Z',
  is_public: false,
  name: 'Image to video',
  revision: 7,
  updated_at: '2026-10-20T00:00:00Z',
  user_id: 'owner-a',
  workflow: { ...RAW_WORKFLOW, meta: { category: 'user', version: '3.0.0' } },
  workflow_id: 'wf-image-to-video',
};

const NO_COPIES: readonly ProjectWorkflowEntry[] = [];

const NOTIFICATIONS = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
const OPEN_ADD_MODELS = vi.fn();
const ADAPTER = {
  notifications: NOTIFICATIONS,
  openAddModels: OPEN_ADD_MODELS,
} as unknown as WorkflowUiAdapter;
const OPEN_DOCUMENT_IN_NEW_PROJECT = vi.fn();
const GRAPH_PREVIEW = {
  openDocumentInNewProject: OPEN_DOCUMENT_IN_NEW_PROJECT,
} as unknown as WorkflowGraphPreviewPort;

describe('WorkflowLibraryDetailPanel', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onClose: () => void;
  let onDeleted: (workflowId: string) => void;
  let onDuplicated: (workflowId: string) => void;
  let onOpen: (item: WorkflowLibraryEntry['item']) => void;
  let onPreview: (workflowId: string) => void;

  /** See `WorkflowLibraryDialog.browser.test.tsx`: keeps Chakra's observer-driven commits inside the act scope. */
  const settleFrame = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

  const panelTree = (
    selected: WorkflowLibraryEntry | null,
    projectWorkflows: readonly ProjectWorkflowEntry[],
    canChangeVisibility = false
  ) => (
    <StrictMode>
      <ChakraProvider value={system}>
        <WorkflowUiProvider adapter={ADAPTER}>
          <GalleryHostProvider host={GALLERY_HOST}>
            <WorkflowGraphPreviewProvider adapter={GRAPH_PREVIEW}>
              <PanesRail
                entry={selected}
                canChangeVisibility={canChangeVisibility}
                projectWorkflows={projectWorkflows}
                onClose={onClose}
                onDeleted={onDeleted}
                onDuplicated={onDuplicated}
                onOpen={onOpen}
                onPreview={onPreview}
              />
            </WorkflowGraphPreviewProvider>
          </GalleryHostProvider>
        </WorkflowUiProvider>
      </ChakraProvider>
    </StrictMode>
  );

  const renderPanel = async (
    selected: WorkflowLibraryEntry | null,
    projectWorkflows: readonly ProjectWorkflowEntry[] = NO_COPIES,
    canChangeVisibility = false
  ) => {
    await act(async () => {
      root.render(panelTree(selected, projectWorkflows, canChangeVisibility));
      await settleFrame();
    });
  };

  const panel = () => document.querySelector<HTMLElement>('[data-workflow-detail]');
  const toggleVisibility = async () => {
    await act(async () => {
      panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.focus();
      await userEvent.keyboard(' ');
      await settleFrame();
    });
  };
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

    // Check open state before clicking the toggle; repeating an open request would otherwise close it.
    if (trigger?.getAttribute('data-state') === 'open') {
      return;
    }

    await act(async () => {
      trigger?.click();
      await settleFrame();
    });
  };

  const menuItem = (value: string) => document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`);

  const clickMenuItem = async (value: string) => {
    await openMenu();

    const item = menuItem(value);
    expect(item, `no menu item "${value}"`).not.toBeNull();

    // Zag selects only the highlighted item, so this takes a real pointer click, which hovers it first.
    await act(() => userEvent.click(item!));
    await act(settleFrame);
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
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    accountLifecycle.invalidate();

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
    queries.getLibraryWorkflowRecord.mockReset().mockResolvedValue({} as Record<string, unknown>);
    queries.getLibraryWorkflowRecordCached.mockClear();
    queries.getLibraryWorkflowRecordCached.mockResolvedValue(RAW_RECORD);
    queries.invalidateWorkflowLibraryCache.mockClear();
    queries.setLibraryWorkflowThumbnail.mockReset().mockResolvedValue(undefined);
    queries.deleteLibraryWorkflowThumbnail.mockReset().mockResolvedValue(undefined);
    queries.setLibraryWorkflowPublic
      .mockReset()
      .mockImplementation((_workflowId, isPublic) => Promise.resolve({ is_public: isPublic }));

    downloadText.mockClear();
    OPEN_DOCUMENT_IN_NEW_PROJECT.mockClear();
    OPEN_ADD_MODELS.mockClear();
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

  it('changes visibility by keyboard, stays server-driven while pending, then reflects the response and invalidates', async () => {
    let resolveChange: (record: { is_public: boolean }) => void = NOOP;
    queries.setLibraryWorkflowPublic.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveChange = resolve;
        })
    );
    const privateEntry = {
      ...TEXT_TO_IMAGE,
      item: { ...TEXT_TO_IMAGE.item, is_public: false, user_id: 'user-a' },
    };

    await renderPanel(privateEntry, NO_COPIES, true);

    const control = panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]');
    expect(panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(false);
    expect(control?.labels?.[0]?.textContent).toBe('Public workflow');
    expect(panel()?.textContent).toContain('Private');
    expect(panel()?.textContent).toContain('Visible to users of this InvokeAI instance.');

    await act(async () => {
      control?.focus();
      await userEvent.keyboard(' ');
      await settleFrame();
    });

    expect(queries.setLibraryWorkflowPublic).toHaveBeenCalledWith('wf-text-to-image', true, expect.anything());
    expect(control?.disabled).toBe(true);
    expect(control?.checked).toBe(false);
    expect(panel()?.textContent).toContain('Private');
    expect(queries.invalidateWorkflowLibraryCache).not.toHaveBeenCalled();

    await act(async () => {
      resolveChange({ is_public: true });
      await settleFrame();
    });

    const updatedControl = panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]');
    expect(updatedControl?.disabled).toBe(false);
    expect(updatedControl?.checked).toBe(true);
    expect(document.activeElement).toBe(updatedControl);
    expect(panel()?.textContent).toContain('Public');
    expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledExactlyOnceWith('wf-text-to-image');
    expect(queries.updateLibraryWorkflow).not.toHaveBeenCalled();
  });

  it('keeps overlapping visibility requests and successful values scoped per workflow', async () => {
    let resolveA: (record: { is_public: boolean }) => void = NOOP;
    let resolveB: (record: { is_public: boolean }) => void = NOOP;
    queries.setLibraryWorkflowPublic.mockImplementation(
      (workflowId) =>
        new Promise((resolve) => {
          if (workflowId === 'wf-a') {
            resolveA = resolve;
          } else {
            resolveB = resolve;
          }
        })
    );
    const privateA = privateWorkflowEntry(TEXT_TO_IMAGE, 'wf-a');
    const privateB = privateWorkflowEntry(IMAGE_TO_VIDEO, 'wf-b');

    await renderPanel(privateA, NO_COPIES, true);
    await toggleVisibility();

    await renderPanel(privateB, NO_COPIES, true);
    await toggleVisibility();

    await act(async () => {
      resolveA({ is_public: true });
      await settleFrame();
    });
    expect(panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.disabled).toBe(true);

    await act(async () => {
      resolveB({ is_public: true });
      await settleFrame();
    });
    expect(panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(true);

    await renderPanel(privateA, NO_COPIES, true);
    expect(panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(true);
    await renderPanel(privateB, NO_COPIES, true);
    expect(panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(true);
  });

  it('reconciles a rejected visibility write from the server and allows another toggle', async () => {
    queries.setLibraryWorkflowPublic.mockRejectedValueOnce(new Error('Request timed out'));
    queries.getLibraryWorkflowRecord.mockResolvedValueOnce({ is_public: true } as Record<string, unknown>);
    const privateEntry = { ...TEXT_TO_IMAGE, item: { ...TEXT_TO_IMAGE.item, is_public: false, user_id: 'user-a' } };

    await renderPanel(privateEntry, NO_COPIES, true);
    await toggleVisibility();

    const publicControl = panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]');
    expect(publicControl?.checked).toBe(true);
    expect(publicControl?.disabled).toBe(false);
    expect(panel()?.textContent).toContain('Public');
    expect(queries.getLibraryWorkflowRecord).toHaveBeenCalledWith('wf-text-to-image', expect.anything());
    expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledWith('wf-text-to-image');

    await toggleVisibility();
    expect(queries.setLibraryWorkflowPublic).toHaveBeenLastCalledWith('wf-text-to-image', false, expect.anything());
  });

  it('shows unknown disabled visibility and retries reconciliation when both write and readback fail', async () => {
    queries.setLibraryWorkflowPublic.mockRejectedValueOnce(new Error('Request timed out'));
    queries.getLibraryWorkflowRecord.mockRejectedValueOnce(new Error('Read failed'));
    await renderPanel(
      { ...TEXT_TO_IMAGE, item: { ...TEXT_TO_IMAGE.item, is_public: false, user_id: 'user-a' } },
      NO_COPIES,
      true
    );
    await toggleVisibility();

    expect(panel()?.textContent).toContain('Visibility unknown');
    expect(panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')).toBeNull();
    expect(panel()?.querySelector('[role="alert"]')?.textContent).toContain('Request timed out');
    expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledExactlyOnceWith('wf-text-to-image');

    queries.getLibraryWorkflowRecord.mockResolvedValueOnce({ is_public: true } as Record<string, unknown>);
    await act(async () => {
      await userEvent.click(buttonWithText('Retry')!);
      await settleFrame();
    });

    expect(panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(true);
    expect(panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.disabled).toBe(false);
    expect(panel()?.textContent).toContain('Public');
    expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledTimes(2);
  });

  it('ignores a visibility readback that resolves after the account lifecycle rotates', async () => {
    let resolveReadback: (record: { is_public: boolean }) => void = NOOP;
    queries.setLibraryWorkflowPublic.mockRejectedValueOnce(new Error('Request timed out'));
    queries.getLibraryWorkflowRecord.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveReadback = resolve;
        })
    );
    try {
      await renderPanel(
        { ...TEXT_TO_IMAGE, item: { ...TEXT_TO_IMAGE.item, is_public: false, user_id: 'user-a' } },
        NO_COPIES,
        true
      );
      await toggleVisibility();
      accountLifecycle.invalidate();

      await act(async () => {
        resolveReadback({ is_public: true });
        await settleFrame();
      });

      expect(panel()?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(false);
      expect(panel()?.querySelector('[data-workflow-visibility="private"]')?.textContent).toBe('Private');
    } finally {
      accountLifecycle.invalidate();
    }
  });

  it('offers Open as the primary action when every requirement is installed', async () => {
    await renderPanel(TEXT_TO_IMAGE);

    expect(requirementStatuses()).toEqual(['installed']);
    expect(buttonWithText('Open')).not.toBeUndefined();
    expect(buttonWithText('Install 1 model')).toBeUndefined();

    await clickButton('Open');

    expect(onOpen).toHaveBeenCalledExactlyOnceWith(TEXT_TO_IMAGE.item);
  });

  it('offers an install under the requirements when starter models can fill the gaps, keeping Open', async () => {
    await renderPanel(IMAGE_TO_VIDEO);

    expect(requirementStatuses()).toEqual(['installable', 'installable', 'installed']);
    expect(document.querySelectorAll('[aria-label="Not installed"]')).toHaveLength(2);
    expect(document.querySelectorAll('[aria-label="Installed"]')).toHaveLength(1);
    const install = buttonWithText('Install 2 models');
    const requirements = document.querySelector('[data-requirement-status]')!;
    expect(requirements.compareDocumentPosition(install!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(buttonWithText('Open')).not.toBeUndefined();
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
    expect(RAW_WORKFLOW.meta).toStrictEqual({ category: 'default', version: '3.0.0' });
    expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledTimes(1);
    expect(onDuplicated).toHaveBeenCalledWith('wf-copy');
    expect(NOTIFICATIONS.success).toHaveBeenCalledWith('Saved a copy under Yours');
  });

  it('forks the cached workflow into a fresh project, carrying its library source and revision', async () => {
    await renderPanel(IMAGE_TO_VIDEO);

    await clickMenuItem('fork-into-project');

    expect(queries.getLibraryWorkflowRecordCached).toHaveBeenCalledWith('wf-image-to-video', expect.anything());
    expect(OPEN_DOCUMENT_IN_NEW_PROJECT).toHaveBeenCalledTimes(1);

    const [document_, label, source] = OPEN_DOCUMENT_IN_NEW_PROJECT.mock.calls[0] as [
      { name: string },
      string,
      unknown,
    ];

    expect(document_.name).toBe('Image to video');
    expect(label).toBe('Image to video');
    expect(source).toEqual({ libraryWorkflowId: 'wf-image-to-video', ownerUserId: 'owner-a', revision: 7 });
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(queries.createLibraryWorkflow).not.toHaveBeenCalled();
  });

  it('carries stable ownership when forking a Shared public workflow', async () => {
    queries.getLibraryWorkflowRecordCached.mockResolvedValueOnce({
      ...RAW_RECORD,
      is_public: true,
      user_id: 'owner-a',
    });
    await renderPanel(IMAGE_TO_VIDEO);

    await clickMenuItem('fork-into-project');

    const [, , source] = OPEN_DOCUMENT_IN_NEW_PROJECT.mock.calls[0] as [unknown, string, unknown];

    expect(source).toEqual({ libraryWorkflowId: 'wf-image-to-video', ownerUserId: 'owner-a', revision: 7 });
  });

  it('renames a user template over the live record at its current revision', async () => {
    queries.getLibraryWorkflowRecord.mockResolvedValueOnce({
      revision: 4,
      workflow: { name: 'Text to image', nodes: [], notes: 'kept' },
      workflow_id: 'wf-text-to-image',
    });

    await renderPanel(TEXT_TO_IMAGE);
    await clickMenuItem('rename');

    const input = document.querySelector<HTMLInputElement>('input[name="renameValue"]');
    expect(input?.value).toBe('Text to image');

    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    await act(async () => {
      setValue?.call(input, 'Portraits');
      input?.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    await act(async () => {
      input?.closest('form')?.querySelector<HTMLButtonElement>('button[type="submit"]')?.click();
      await settleFrame();
      await settleFrame();
    });

    expect(queries.updateLibraryWorkflow).toHaveBeenCalledWith(
      'wf-text-to-image',
      { name: 'Portraits', nodes: [], notes: 'kept' },
      expect.objectContaining({ expectedRevision: 4 })
    );
    expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledWith('wf-text-to-image');
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
    expect(onDeleted).toHaveBeenCalledExactlyOnceWith('wf-text-to-image');
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

  it('links only the rows the account is missing, and takes them to Add Models', async () => {
    await renderPanel(IMAGE_TO_VIDEO);

    expect(requirementStatuses()).toEqual(['installable', 'installable', 'installed']);
    expect(document.querySelectorAll('[data-requirement-link]')).toHaveLength(2);
    expect(requirementRows()[2]?.querySelector('[data-requirement-link]')).toBeNull();

    const link = document.querySelector<HTMLElement>('[data-requirement-link]');
    expect(link?.textContent).toBe('Wan 2.2 I2V A14B');
    await act(() => link?.focus());
    await expect
      .poll(() => document.querySelector('[role="tooltip"]')?.textContent, { timeout: 3_000 })
      .toBe('Find Wan 2.2 I2V A14B in the model manager');

    await act(async () => {
      link?.click();
      await settleFrame();
    });

    expect(OPEN_ADD_MODELS).toHaveBeenCalledWith('Wan 2.2 I2V A14B');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  describe('thumbnail', () => {
    // The library dialog preloads the preview chunk as it opens; a cold import can outlast `waitFor` under load.
    beforeAll(() => loadGraphPreview());

    const buttonNamed = (name: string) =>
      [...(panel()?.querySelectorAll('button') ?? [])].find(
        (candidate) => (candidate.getAttribute('aria-label') ?? candidate.textContent ?? '').trim() === name
      );
    const choose = () => buttonNamed('Choose thumbnail');
    const pick = async (item: Record<string, unknown>) => {
      pickedGalleryItem.current = item;
      await act(async () => {
        document.querySelector<HTMLElement>('[data-gallery-pick]')?.click();
        await settleFrame();
      });
    };
    const GIF_ITEM = {
      kind: 'image',
      name: 'picked.gif',
      thumbnailUrl: 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==',
    };

    it("is chosen from the preview's corner only on the account's own templates, with Remove once one is set", async () => {
      await renderPanel(DEFAULT_CATEGORY);
      expect(choose()).toBeUndefined();

      await renderPanel(entry({ name: 'Plain', workflow_id: 'wf-plain' }, readyEnrichment(SDXL_REQUIREMENTS)));
      expect(choose()).not.toBeUndefined();
      expect(buttonNamed('Remove thumbnail')).toBeUndefined();
      expect(buttonNamed('Upload')).toBeUndefined();

      await renderPanel(TEXT_TO_IMAGE);
      expect(choose()).not.toBeUndefined();
      expect(buttonNamed('Remove thumbnail')).not.toBeUndefined();
    });

    it('stores a picked gallery image, holding the slot busy until the library answers', async () => {
      let answer: () => void = NOOP;
      queries.setLibraryWorkflowThumbnail.mockImplementation(
        () =>
          new Promise<void>((resolve) => {
            answer = resolve;
          })
      );

      await renderPanel(TEXT_TO_IMAGE);
      const status = panel()?.querySelector('[role="status"]');
      expect(status?.textContent).toBe('');
      await pick(GIF_ITEM);
      await vi.waitFor(() => expect(queries.setLibraryWorkflowThumbnail).toHaveBeenCalled());

      const [workflowId, image] = queries.setLibraryWorkflowThumbnail.mock.calls[0]!;
      expect(workflowId).toBe('wf-text-to-image');
      expect(image.type).toBe('image/gif');
      expect(panel()?.querySelector('[role="status"]')).toBe(status);
      expect(status?.textContent).toBe('Updating thumbnail…');
      expect(panel()?.querySelector('[aria-busy="true"]')).not.toBeNull();
      expect(choose()?.getAttribute('aria-disabled')).toBe('true');
      expect(buttonNamed('Remove thumbnail')?.getAttribute('aria-disabled')).toBe('true');
      expect(queries.invalidateWorkflowLibraryCache).not.toHaveBeenCalled();

      await act(async () => {
        answer();
        await settleFrame();
      });

      expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledWith('wf-text-to-image');
      expect(panel()?.querySelector('[aria-busy="true"]')).toBeNull();
      expect(choose()?.hasAttribute('aria-disabled')).toBe(false);
      expect(status?.textContent).toBe('Thumbnail updated');
    });

    it('does not carry a busy upload over to the next selected workflow', async () => {
      queries.setLibraryWorkflowThumbnail.mockImplementation(() => new Promise<void>(NOOP));

      await renderPanel(TEXT_TO_IMAGE);
      await pick(GIF_ITEM);
      await vi.waitFor(() => expect(panel()?.querySelector('[aria-busy="true"]')).not.toBeNull());

      await renderPanel(IMAGE_TO_VIDEO);
      expect(panel()?.querySelector('[aria-busy="true"]')).toBeNull();
      expect(panel()?.querySelector('[role="status"]')?.textContent).toBe('');
    });

    it('snapshots the graph once its document has loaded, storing the captured image', async () => {
      const snapshot = () => buttonNamed('Snapshot the graph');

      await renderPanel(entry({ name: 'Pending', workflow_id: 'wf-pending' }, { status: 'pending' }));
      expect(snapshot()?.getAttribute('aria-disabled')).toBe('true');

      await renderPanel(TEXT_TO_IMAGE);
      expect(snapshot()?.hasAttribute('aria-disabled')).toBe(false);
      await act(async () => {
        snapshot()?.click();
        await settleFrame();
      });

      await vi.waitFor(() =>
        expect(queries.setLibraryWorkflowThumbnail).toHaveBeenCalledWith(
          'wf-text-to-image',
          SNAPSHOT_IMAGE,
          expect.anything()
        )
      );
      await vi.waitFor(() => expect(queries.invalidateWorkflowLibraryCache).toHaveBeenCalledWith('wf-text-to-image'));
    });
  });
});
