import type { StarterModel } from '@features/models';
import type { WorkflowModelRequirement } from '@features/workflow/core/modelRequirements';
import type {
  InvocationTemplate,
  InvocationTemplatesSnapshot,
  ProjectGraphState,
  ProjectWorkflowEntry,
} from '@features/workflow/core/types';
import type * as libraryBrowseStoreModule from '@features/workflow/data/libraryBrowseStore';
import type {
  WorkflowLibraryBrowseSnapshot,
  WorkflowLibraryEntry,
  WorkflowLibraryEntryEnrichment,
} from '@features/workflow/data/libraryBrowseStore';
import type * as workflowQueriesModule from '@features/workflow/queries';
import type {
  WorkflowGraphPreviewPort,
  WorkflowPreferences,
  WorkflowReadPort,
  WorkflowUiAdapter,
} from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { GalleryHostProvider, type GalleryHost } from '@features/gallery/picker';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import {
  openWorkflowLibraryAtProjectWorkflow,
  setWorkflowLibraryOpen,
  setWorkflowLibraryTab,
  workflowUiStore,
} from '@features/workflow/ui/workflowUiStore';
import { buildInvocationNode, createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { settleAnimations } from '@platform/browser/settleAnimations.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';
// Warm the mocked lazy module to avoid first-compile timing variability in preview wiring tests.
import '@features/workflow/ui/graph-preview/GraphPreviewDialog';

import { LibraryCopyChoiceHost } from './LibraryCopyChoiceDialog';
import { keepCardMenuOpenForRetarget } from './WorkflowLibraryCard';
import { WorkflowLibraryDialog } from './WorkflowLibraryDialog';

/** In the app the workbench derives this from its Gallery adapter; the library's thumbnail slot reads it. */
const GALLERY_HOST: GalleryHost = {
  galleryValues: {},
  notifications: { add: vi.fn(), reportError: vi.fn() },
  projectName: '',
};

// Provide the fixture node's reactive template snapshot so preview compilation can run without backend schema
// loading.
const PREVIEW_NODE_TEMPLATE: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {},
  nodePack: 'invokeai',
  outputs: {},
  outputType: 'integer_output',
  tags: [],
  title: 'integer',
  type: 'integer',
  useCache: true,
  version: '1.0.0',
};

const TEMPLATES_SNAPSHOT: InvocationTemplatesSnapshot = {
  error: null,
  status: 'loaded',
  templates: { integer: PREVIEW_NODE_TEMPLATE },
};

const identity = vi.hoisted(() => ({
  session: { multiuserEnabled: false, phase: 'ready', user: null } as {
    multiuserEnabled: boolean;
    phase: 'ready';
    user: { user_id: string } | null;
  },
}));

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationTemplatesSnapshot: () => TEMPLATES_SNAPSHOT,
}));

/** The library templates the project rail reads by id; an absent id fails as a missing record does. */
const templateRecords = vi.hoisted(() => new Map<string, { name: string; revision: number }>());

/** The library writes a card menu's Rename and Delete make; each answers as the server would. */
const libraryWrites = vi.hoisted(() => ({
  cached: vi.fn((workflowId: string, _options: { expectedRevision: number }) =>
    Promise.resolve({ id: workflowId, meta: { category: 'user' }, name: workflowId })
  ),
  create: vi.fn((_workflow: { name: string }, _signal?: AbortSignal) => Promise.resolve('wf-landscape-copy')),
  delete: vi.fn((_workflowId: string, _signal?: AbortSignal) => Promise.resolve()),
  record: vi.fn((workflowId: string, _signal?: AbortSignal) =>
    Promise.resolve({ revision: 3, workflow: { id: workflowId, name: workflowId }, workflow_id: workflowId })
  ),
  update: vi.fn((_workflowId: string, _workflow: { name: string }, _options: { expectedRevision: number }) =>
    Promise.resolve()
  ),
}));

vi.mock('@features/workflow/queries', async (importOriginal) => {
  const original = await importOriginal<typeof workflowQueriesModule>();

  return {
    ...original,
    createLibraryWorkflow: libraryWrites.create,
    deleteLibraryWorkflow: libraryWrites.delete,
    getLibraryWorkflowCached: libraryWrites.cached,
    getLibraryWorkflowRecord: libraryWrites.record,
    invalidateWorkflowLibraryCache: vi.fn(),
    updateLibraryWorkflow: libraryWrites.update,
    savedWorkflowDetailQueryOptions: (workflowId: string) => ({
      ...original.savedWorkflowDetailQueryOptions(workflowId),
      queryFn: () => {
        const record = templateRecords.get(workflowId);

        return record
          ? Promise.resolve(record as unknown as workflowQueriesModule.WorkflowRecordDTO)
          : Promise.reject(new Error('not found'));
      },
    }),
  };
});

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuthSession: () => identity.session,
}));

// Stub preview rendering while exposing compiled graph, hideInvoke, and close behavior owned by dialog wiring.
vi.mock('@features/workflow/ui/graph-preview/GraphPreviewDialog', () => ({
  GraphPreviewDialog: ({
    graphId,
    hideInvoke,
    isOpen,
    source,
    sourceLabel,
    onExitComplete,
    onOpenChange,
  }: {
    graphId: string;
    hideInvoke?: boolean;
    isOpen: boolean;
    source: { graph: { nodes: { id: string; type: string }[] } | null };
    sourceLabel: string;
    onExitComplete?: () => void;
    onOpenChange: (isOpen: boolean) => void;
  }) => (
    <div
      data-graph-id={graphId}
      data-hide-invoke={String(hideInvoke)}
      data-preview-dialog
      data-preview-open={String(isOpen)}
      data-source-label={sourceLabel}
    >
      {(source.graph?.nodes ?? []).map((node) => node.type).join(',')}
      <button onClick={() => onOpenChange(false)}>Close preview</button>
      {/* Stands in for Ark's presence machine: the real dialog reports the end
          of its close transition, which is what releases the mount. */}
      <button onClick={() => onExitComplete?.()}>Finish preview exit</button>
    </div>
  ),
}));

// Drive a real external browse store to retain subscription and selector behavior while spying on commands.
const browse = vi.hoisted(() => ({
  ensureWorkflowLibraryBrowseLoaded: vi.fn((_opening: Partial<WorkflowLibraryBrowseSnapshot['filter']>) =>
    Promise.resolve()
  ),
  loadWorkflowLibraryPlaceTotals: vi.fn((_places: readonly string[]) => Promise.resolve()),
  // Assigned by the module factory below, which owns the store instance.
  getSnapshot: null as null | (() => WorkflowLibraryBrowseSnapshot),
  loadWorkflowLibraryRange: vi.fn((_start: number, _end: number) => Promise.resolve(true)),
  setSnapshot: null as null | ((snapshot: WorkflowLibraryBrowseSnapshot) => void),
  setWorkflowLibraryBrowseFilter: vi.fn(),
}));

vi.mock('@features/workflow/data/libraryBrowseStore', async (original) => {
  const {
    getWorkflowLibraryOrder,
    getWorkflowLibrarySortDefaultDirection,
    selectWorkflowLibraryEntry,
    WORKFLOW_LIBRARY_ENRICHMENT_DEMAND,
    WORKFLOW_LIBRARY_METADATA_WINDOW,
  } = await original<typeof libraryBrowseStoreModule>();
  const { createExternalStore } = await import('@platform/state/externalStore');
  const store = createExternalStore<WorkflowLibraryBrowseSnapshot>({
    entries: [],
    error: null,
    filter: { direction: 'desc', place: 'user', search: '', sort: 'updated', tags: [] },
    page: 0,
    pages: 0,
    status: 'idle',
    tagCounts: [],
    total: 0,
    placeTotals: {},
  });

  browse.getSnapshot = store.getSnapshot;
  browse.setSnapshot = store.setSnapshot;

  return {
    getWorkflowLibraryOrder,
    getWorkflowLibrarySortDefaultDirection,
    holdWorkflowLibraryBrowse: () => () => {},
    selectWorkflowLibraryEntry,
    WORKFLOW_LIBRARY_ENRICHMENT_DEMAND,
    WORKFLOW_LIBRARY_METADATA_WINDOW,
    ensureWorkflowLibraryBrowseLoaded: browse.ensureWorkflowLibraryBrowseLoaded,
    loadWorkflowLibraryPlaceTotals: browse.loadWorkflowLibraryPlaceTotals,
    getWorkflowLibraryBrowseSnapshot: store.getSnapshot,
    loadWorkflowLibraryRange: browse.loadWorkflowLibraryRange,
    refreshWorkflowLibraryBrowse: vi.fn(() => Promise.resolve()),
    rememberWorkflowLibraryScroll: vi.fn(),
    setWorkflowLibraryVisibleDemand: vi.fn(),
    setWorkflowLibrarySelectedDemand: vi.fn(),
    setWorkflowLibraryBrowseFilter: browse.setWorkflowLibraryBrowseFilter,
    useWorkflowLibraryBrowseSelector: store.useSelector,
  };
});

// Test opener invocation and busy-state wiring here; load sequencing has separate coverage.
const loader = vi.hoisted(() => ({
  open: vi.fn((_item: unknown, _mode: 'first-copy' | 'add-copy') => Promise.resolve()),
  phase: { current: 'idle' as 'applying' | 'fetching' | 'idle' },
  replace: vi.fn((_item: unknown, _workflowId: string) => Promise.resolve()),
  resume: vi.fn((_workflowId: string) => {}),
}));

vi.mock('./useOpenLibraryWorkflow', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useOpenLibraryWorkflow: () => ({
    loadPhase: loader.phase.current,
    open: loader.open,
    replace: loader.replace,
    resume: loader.resume,
  }),
}));

// Use fixed model-store data so missing-model badges test dialog wiring rather than live catalogs.
const FLUX_STARTER: StarterModel = {
  base: 'flux',
  description: 'FLUX.1 dev',
  is_installed: false,
  name: 'FLUX.1 dev',
  source: 'https://models.test/flux-dev',
  type: 'main',
};

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: vi.fn(() => Promise.resolve()),
  ensureStartersLoaded: vi.fn(),
  useActiveInstallSources: () => new Set<string>(),
  useInstallActions: () => ({ install: vi.fn(), installMany: vi.fn(), pendingSources: new Set() }),
  useModelsSelector: (selector: (snapshot: unknown) => unknown) => selector({ models: [] }),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ response: { starter_models: [FLUX_STARTER] } }),
}));

// Provide English/plural strings without booting the HTTP-backed i18n client.
const TRANSLATIONS: Record<string, string> = {
  'workflowLibrary.openProjectWorkflowHint': 'Makes it the active workflow',
  'workflowLibrary.renameProjectWorkflowHint': 'Changes its name in this project; a linked template keeps its own',
  'workflowLibrary.duplicateProjectWorkflowHint': 'A copy beside it in this project',
  'workflowLibrary.saveToLibraryHint': 'Creates a new template from this workflow',
  'workflowLibrary.updateTemplateHint': 'Writes this workflow over its library template',
  'workflowLibrary.removeHint': 'Removes it from this project only',
  'workflowLibrary.renameTemplateHint': "Changes the template's name in the library",
  'workflowLibrary.duplicateHint': 'Saves a copy in My library, original untouched',
  'workflowLibrary.downloadJsonHint': 'For bug reports and sharing',
  'workflowLibrary.forkIntoProjectHint': 'Copies it into a new project',
  'workflowLibrary.deleteHint': 'Removes it from your library',

  'common.cancel': 'Cancel',
  'common.close': 'Close',
  'workflowLibrary.activeWorkflow': 'Active',
  'workflowLibrary.addAnotherCopy': 'Add another copy',
  'workflowLibrary.addAnotherCopyHint': 'A second, independent copy in this project',
  'workflowLibrary.chooseProjectCopy': 'Open which copy?',
  'workflowLibrary.applying': 'Applying workflow…',
  'workflowLibrary.delete': 'Delete',
  'workflowLibrary.deleteConfirmBody': 'Delete "{{name}}" from the workflow library? This cannot be undone.',
  'workflowLibrary.deleteConfirmTitle': 'Delete workflow',
  'workflowLibrary.downloadJson': 'Download JSON',
  'workflowLibrary.duplicate': 'Duplicate',
  'workflowLibrary.duplicateName': '{{name}} copy',
  'workflowLibrary.empty': 'No workflows match these filters.',
  'workflowLibrary.fetching': 'Fetching workflow…',
  'workflowLibrary.forkIntoProject': 'Fork into new project',
  'workflowLibrary.installModels_one': 'Install 1 model',
  'workflowLibrary.installModels_other': 'Install {{count}} models',
  'workflowLibrary.loading': 'Loading workflows…',
  'workflowLibrary.moreActions': 'More actions',
  'workflowLibrary.newWorkflow': 'New workflow',
  'workflowLibrary.nodeCount_one': '{{count}} node',
  'workflowLibrary.nodeCount_other': '{{count}} nodes',
  'workflowLibrary.notRunYet': 'Not run yet',
  'workflowLibrary.open': 'Open',
  'workflowLibrary.openHint': 'Adds a copy to this project',
  'workflowLibrary.openWithEllipsis': 'Open…',
  'workflowLibrary.previewGraph': 'Preview graph',
  'workflowLibrary.private': 'Private',
  'workflowLibrary.public': 'Public',
  'workflowLibrary.publicWorkflow': 'Public workflow',
  'workflowLibrary.visibilityUpdateFailed': 'Failed to update workflow visibility',
  'workflowLibrary.remove': 'Remove',
  'workflowLibrary.removeConfirmBody':
    'Remove "{{name}}" from this project? Its library template, if any, is not affected.',
  'workflowLibrary.removeConfirmTitle': 'Remove workflow',
  'workflowLibrary.removeWithEllipsis': 'Remove…',
  'workflowLibrary.rename': 'Rename',
  'workflowLibrary.renameTitle': 'Rename workflow',
  'workflowLibrary.renameWithEllipsis': 'Rename…',
  'workflowLibrary.requirementInstallable': 'Not installed',
  'workflowLibrary.requires': 'Requires',
  'workflowLibrary.sampleOutput': 'Sample output',
  'workflowLibrary.saveToLibraryWithEllipsis': 'Save to library…',
  'workflowLibrary.searchPlaceholder': 'Search names, tags, or descriptions',
  'workflowLibrary.sortCreated': 'Date created',
  'workflowLibrary.sortLabel': 'Sort by',
  'workflowLibrary.sortName': 'Name',
  'workflowLibrary.sortOpened': 'Last opened',
  'workflowLibrary.sortUpdated': 'Last updated',
  'workflowLibrary.sourceBundled': 'From a bundled template (read-only)',
  'workflowLibrary.sourceNone': 'Not linked to a library template',
  'workflowLibrary.sourceRevision': 'Linked to library template at revision {{revision}}',
  'workflowLibrary.thumbnailFit': 'Thumbnail display',
  'workflowLibrary.thumbnailFitContain': 'Fit',
  'workflowLibrary.thumbnailFitCover': 'Fill',
  'workflowLibrary.thisProjectHint':
    'Workflows saved with this project. Edits autosave here; the library changes only when you save to it.',
  'workflowLibrary.title': 'Workflows',
  'workflowLibrary.cardSize': 'Card size',
  'workflowLibrary.cardSizeLarge': 'Large cards',
  'workflowLibrary.cardSizeSmall': 'Small cards',
  'workflowLibrary.clearTags': 'Clear',
  'workflowLibrary.emptyRecent': 'Nothing opened yet. Workflows you open from My library or Templates appear here.',
  'workflowLibrary.emptyShared': 'Nothing shared yet. Public workflows on this server appear here.',
  'workflowLibrary.emptyUser': 'No saved workflows yet. Use Save to library… in the editor to keep one here.',
  'workflowLibrary.placeDefault': 'Templates',
  'workflowLibrary.placeProject': 'In this project',
  'workflowLibrary.placeRecent': 'Recent',
  'workflowLibrary.placeShared': 'Shared with me',
  'workflowLibrary.placeUser': 'My library',
  'workflowLibrary.places': 'Places',
  'workflowLibrary.resultCount_one': '{{count}} workflow',
  'workflowLibrary.resultCount_other': '{{count}} workflows',
  'workflowLibrary.resultCountWithTags_one': '{{count}} workflow · {{tags}}',
  'workflowLibrary.resultCountWithTags_other': '{{count}} workflows · {{tags}}',
  'workflowLibrary.sortAToZ': 'A to Z',
  'workflowLibrary.sortControl': 'Sort: {{sort}}, {{direction}}',
  'workflowLibrary.sortFirstToLast': 'First to last',
  'workflowLibrary.sortLastRun': 'Last run',
  'workflowLibrary.sortLastToFirst': 'Last to first',
  'workflowLibrary.sortNewestFirst': 'Newest first',
  'workflowLibrary.sortOldestFirst': 'Oldest first',
  'workflowLibrary.sortProjectOrder': 'Project order',
  'workflowLibrary.sortZToA': 'Z to A',
  'workflowLibrary.tags': 'Tags',
  'workflowLibrary.tagsAnyHint': 'Showing workflows with any checked tag',
  'workflowLibrary.viewOptions': 'View options',
  'workflowLibrary.layoutGroup': 'View',
  'workflowLibrary.layoutGrid': 'Grid',
  'workflowLibrary.layoutList': 'List',
  'workflowLibrary.untitled': 'Untitled Workflow',
  'workflowLibrary.updateTemplate': 'Update library template',
  'workflowLibrary.workflowName': 'Workflow name',
  'workflowLibrary.activeRanWhen': 'Active workflow · ran {{when}}',
  'workflowLibrary.activeWorkflowLine': 'Active workflow',
  'workflowLibrary.callable': 'Callable',
  'workflowLibrary.factCreated': 'Created',
  'workflowLibrary.factEdited': 'Edited',
  'workflowLibrary.factInProject': 'In this project',
  'workflowLibrary.factLastRun': 'Last run',
  'workflowLibrary.factUpdated': 'Updated',
  'workflowLibrary.inProject': 'In project',
  'workflowLibrary.linkedTemplate': 'Linked template',
  'workflowLibrary.modelsToInstall_one': '{{count}} model to install',
  'workflowLibrary.modelsToInstall_other': '{{count}} models to install',
  'workflowLibrary.nodePacksMissing_one': '{{count}} node pack missing',
  'workflowLibrary.nodePacksMissing_other': '{{count}} node packs missing',
  'workflowLibrary.openedWhen': 'Opened {{when}}',
  'workflowLibrary.projectCopies_one': '{{count}} copy',
  'workflowLibrary.projectCopies_other': '{{count}} copies',
  'workflowLibrary.projectCopiesNone': 'None',
  'workflowLibrary.projectCopyRevision': '1 copy · revision {{revision}}',
  'workflowLibrary.ranWhen': 'Ran {{when}}',
  'workflowLibrary.readyToRun': 'Ready to run',
  'workflowLibrary.saveToTemplate': 'Save to template',
  'workflowLibrary.templateNewer': 'Template is newer',
  'workflowLibrary.templateRevision': 'Library template is at revision {{revision}}',
  'workflowLibrary.updateFromTemplate': 'Update from template',
  'workflowLibrary.updateFromTemplateBody': 'Replace "{{name}}" with the library template’s current version?',
  'workflowLibrary.updateFromTemplateConfirm': 'Update',
  'workflowLibrary.updatedFromTemplate': 'Updated from template',
  'workflowLibrary.whenAtRevision': '{{when}} · revision {{revision}}',
  'workflowLibrary.detailsAndRequirements': 'Details and requirements',
  'workflowLibrary.hideDetails': 'Hide details',
  'workflowLibrary.sortAndViewOptions': 'Sort and view options',
  'workflowLibrary.sortDirectionGroup': 'Direction',
  'workflowLibrary.missingNodePackNames_one': 'Missing node pack: {{packs}}',
  'workflowLibrary.missingNodePackNames_other': 'Missing node packs: {{packs}}',
  'workflowLibrary.sourceNamedRevision': 'Linked to {{name}}, revision {{revision}}',
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

const EMPTY_DOCUMENT = createProjectGraph('library-fixture');

const PREVIEW_DOCUMENT: ProjectGraphState = projectGraphReducer(createProjectGraph('preview-fixture'), {
  node: buildInvocationNode(PREVIEW_NODE_TEMPLATE, { x: 0, y: 0 }),
  type: 'addNode',
});

const NO_REQUIREMENTS: readonly WorkflowModelRequirement[] = [];

const readyEnrichment = (
  nodeCount: number,
  primaryBase: string | null,
  requirements: readonly WorkflowModelRequirement[] = NO_REQUIREMENTS,
  document: ProjectGraphState = EMPTY_DOCUMENT
): WorkflowLibraryEntryEnrichment => ({
  document,
  nodeCount,
  requirements: { primaryBase, requirements },
  status: 'ready',
});

const entry = (
  workflowId: string,
  name: string,
  enrichment: WorkflowLibraryEntryEnrichment,
  extras: { tags?: readonly string[]; thumbnailUrl?: string } = {}
): WorkflowLibraryEntry => ({
  enrichment,
  item: {
    category: 'user',
    description: `${name} description`,
    name,
    revision: 1,
    thumbnail_url: extras.thumbnailUrl ?? null,
    workflow_id: workflowId,
  },
  tags: extras.tags ?? [],
});

const PORTRAIT = entry('wf-portrait', 'Portrait Studio', readyEnrichment(12, 'sdxl'), {
  tags: ['portrait'],
  thumbnailUrl: 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==',
});
const NON_OWNER_SHARED = { ...PORTRAIT, item: { ...PORTRAIT.item, user_id: 'user-b' } };
const OWNER_PRIVATE = { ...PORTRAIT, item: { ...PORTRAIT.item, is_public: false, user_id: 'user-a' } };
const OWNER_PUBLIC = { ...PORTRAIT, item: { ...PORTRAIT.item, is_public: true, user_id: 'user-a' } };
// One absent requirement with a matching starter should show one model to install.
const LANDSCAPE = entry(
  'wf-landscape',
  'Landscape Pass',
  readyEnrichment(7, 'flux', [{ base: 'flux', kind: 'slot', label: 'FLUX checkpoint', modelType: 'main' }]),
  { tags: ['landscape'] }
);
const SKETCH = entry('wf-sketch', 'Sketch To Render', { status: 'pending' });
const UPSCALE = entry(
  'wf-upscale',
  'Upscale Chain',
  { message: 'unreadable', status: 'error' },
  {
    tags: ['upscale'],
  }
);
const PREVIEW_FIXTURE = entry(
  'wf-preview-fixture',
  'Preview Fixture',
  readyEnrichment(1, null, NO_REQUIREMENTS, PREVIEW_DOCUMENT)
);

/** ~80 characters with no break opportunity the rail could exploit. */
const LONG_NAME = 'Cinematic Portrait Restoration And Upscaling Pipeline With Refiner And Face Detail';
const LONG_NAMED = entry('wf-long-name', LONG_NAME, readyEnrichment(3, 'sdxl'));

const LOADED_SNAPSHOT: WorkflowLibraryBrowseSnapshot = {
  entries: [PORTRAIT, LANDSCAPE, SKETCH, UPSCALE],
  error: null,
  filter: { direction: 'desc', place: 'user', search: '', sort: 'updated', tags: [] },
  page: 0,
  pages: 2,
  status: 'loaded',
  tagCounts: [
    { count: 2, tag: 'portrait' },
    { count: 1, tag: 'upscale' },
  ],
  total: 24,
  placeTotals: { user: 4 },
};

const withSnapshot = (patch: Partial<WorkflowLibraryBrowseSnapshot>): WorkflowLibraryBrowseSnapshot => ({
  ...LOADED_SNAPSHOT,
  ...patch,
});

const createMutablePort = <Snapshot,>(initialSnapshot: Snapshot) => {
  let snapshot = initialSnapshot;
  const listeners = new Set<() => void>();
  const port: WorkflowReadPort<Snapshot> = {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return {
    port,
    setSnapshot: (next: Snapshot) => {
      snapshot = next;
      for (const listener of listeners) {
        listener();
      }
    },
  };
};

const PROJECT_ID = 'project-1';

/** The project's own collection: an unlinked active workflow, a bundled copy, and a copy of a user template. */
const ALPHA: ProjectWorkflowEntry = { document: { ...createProjectGraph('wf-alpha'), name: 'Alpha' } };
const BUNDLED_COPY: ProjectWorkflowEntry = {
  document: { ...createProjectGraph('wf-bundled'), name: 'Bundled copy' },
  source: { libraryWorkflowId: 'default_text_to_image', revision: 1 },
};
const USER_COPY: ProjectWorkflowEntry = {
  document: { ...PREVIEW_DOCUMENT, id: 'wf-user', name: 'User copy' },
  source: { libraryWorkflowId: 'wf-portrait', ownerUserId: 'user-a', revision: 4 },
};
const PROJECT_WORKFLOWS: readonly ProjectWorkflowEntry[] = [ALPHA, BUNDLED_COPY, USER_COPY];

const projectSnapshot = (workflows: readonly ProjectWorkflowEntry[], activeWorkflowId = 'wf-alpha') => {
  const activeWorkflow = workflows.find((entry) => entry.document.id === activeWorkflowId) ?? workflows[0]!;

  return {
    activeWorkflow,
    activeWorkflowId: activeWorkflow.document.id,
    galleryValues: {},
    id: PROJECT_ID,
    isWorkflowRunning: false,
    projectGraph: activeWorkflow.document,
    workflowValues: {},
    workflows,
  };
};

const project = createMutablePort(projectSnapshot(PROJECT_WORKFLOWS));
const COMMANDS = {
  addWorkflow: vi.fn(() => 'wf-added'),
  createWorkflow: vi.fn(() => 'wf-new'),
  duplicateWorkflow: vi.fn((_workflowId: string, _copyName: string) => 'wf-copy'),
  editGraph: vi.fn(),
  redo: vi.fn(),
  removeWorkflow: vi.fn(),
  renameWorkflow: vi.fn(),
  selectWorkflow: vi.fn(),
  setWorkflowSource: vi.fn(),
  undo: vi.fn(),
};
/** Only the library's own preferences matter here; saving one republishes it, as the account store does. */
const LIBRARY_PREFERENCES = {
  workflowLibraryCardSize: 'small',
  workflowLibraryLayout: 'grid',
  workflowLibrarySort: 'updated',
  workflowLibrarySortDirection: 'desc',
  workflowLibraryThumbnailFit: 'cover',
} as const;
const preferences = createMutablePort<WorkflowPreferences>(LIBRARY_PREFERENCES as unknown as WorkflowPreferences);
const setLibraryPreferences = vi.fn((patch: Partial<WorkflowPreferences>) =>
  preferences.setSnapshot({ ...preferences.port.getSnapshot(), ...patch })
);
const UI_ADAPTER = {
  commands: COMMANDS,
  preferences: preferences.port,
  setLibraryPreferences,
  getProjectGraph: () => project.port.getSnapshot().projectGraph,
  notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
  openAddModels: vi.fn(),
  project: project.port,
} as unknown as WorkflowUiAdapter;
const GRAPH_PREVIEW = { openDocumentInNewProject: vi.fn() } as unknown as WorkflowGraphPreviewPort;
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

/** The library as the workflow dialog host mounts it: open while the UI store says so. */
const StoreBoundLibraryDialog = () => {
  const isOpen = workflowUiStore.useSelector((snapshot) => snapshot.isLibraryOpen);

  return <WorkflowLibraryDialog isOpen={isOpen} onOpenChange={setWorkflowLibraryOpen} />;
};

describe('WorkflowLibraryDialog', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onOpenChange: (isOpen: boolean) => void;

  /** Await deferred SegmentGroup observer updates inside the render act scope, including StrictMode remounts. */
  const settleFrame = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

  const renderDialog = async (isOpen = true) => {
    await act(async () => {
      root.render(
        <StrictMode>
          <QueryClientProvider client={queryClient}>
            <ChakraProvider value={system}>
              <WorkflowUiProvider adapter={UI_ADAPTER}>
                <GalleryHostProvider host={GALLERY_HOST}>
                  <WorkflowGraphPreviewProvider adapter={GRAPH_PREVIEW}>
                    <WorkflowLibraryDialog isOpen={isOpen} onOpenChange={onOpenChange} />
                    <LibraryCopyChoiceHost />
                  </WorkflowGraphPreviewProvider>
                </GalleryHostProvider>
              </WorkflowUiProvider>
            </ChakraProvider>
          </QueryClientProvider>
        </StrictMode>
      );
      await settleFrame();
    });
  };

  const openWith = async (snapshot: WorkflowLibraryBrowseSnapshot) => {
    await act(() => browse.setSnapshot?.(snapshot));
    await renderDialog();
  };

  /** Drains the microtask + timer queue the store probe and debounce settle on. */
  const wait = (ms: number) =>
    act(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, ms);
      });
    });

  /** Poll lazy preview resolution inside act to avoid timing-dependent waits and suspended-resource warnings. */
  const waitForPreviewDialog = () =>
    act(async () => {
      await vi.waitFor(
        () => {
          expect(document.querySelector('[data-preview-dialog]')).not.toBeNull();
        },
        // Allow time for Vite's dynamic-import round trip even when the module is warm.
        { timeout: 2000 }
      );
    });

  const cards = () => [...document.querySelectorAll<HTMLElement>('[data-workflow-card]')];
  const card = (workflowId: string) => document.querySelector<HTMLElement>(`[data-workflow-card="${workflowId}"]`);
  const buttonWithText = (text: string) =>
    [...document.querySelectorAll('button')].find((candidate) => (candidate.textContent ?? '') === text);

  /** A pointer press on a tile's menu button: pointerdown (an outside press for an open menu), then the click. */
  const pressTileMenuButton = async (button: HTMLElement | null) => {
    expect(button).not.toBeNull();
    await act(async () => {
      button?.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerType: 'mouse' }));
      await settleFrame();
      button?.click();
      await settleFrame();
      await settleFrame();
    });
  };

  const clickText = async (text: string) => {
    const button = buttonWithText(text);
    expect(button).not.toBeUndefined();

    await act(() => button?.click());
  };

  const placeButton = (value: string) => document.querySelector<HTMLButtonElement>(`[data-library-place="${value}"]`);
  const tagCheckbox = (tag: string) =>
    document.querySelector<HTMLInputElement>(`[data-library-tag="${tag}"] input[type="checkbox"]`);
  const clickTag = async (tag: string) => {
    const label = document.querySelector<HTMLElement>(`[data-library-tag="${tag}"]`);
    expect(label).not.toBeNull();

    await act(() => userEvent.click(label!));
  };
  const clickPlace = async (value: string) => {
    const place = placeButton(value);
    expect(place).not.toBeNull();

    await act(async () => {
      place?.click();
      await settleFrame();
    });
  };

  /** The dialog sizes itself to the window, so layout tests set it; the runner's own size returns afterwards. */
  const resizeViewport = async (width: number, height: number) => {
    const previous = { height: window.innerHeight, width: window.innerWidth };

    await page.viewport(width, height);
    onTestFinished(() => page.viewport(previous.width, previous.height));
  };

  const gridViewport = () => document.querySelector<HTMLElement>('[role="region"][aria-label="Workflows"]');
  /** Whether an element takes part in the layout, and so the tab order: each layout hides the other's controls. */
  const isShown = (element: Element | null | undefined): element is HTMLElement =>
    element instanceof HTMLElement && element.getClientRects().length > 0;

  const expectContained = (element: HTMLElement | Range, container: HTMLElement) => {
    const box = element.getBoundingClientRect();
    const bounds = container.getBoundingClientRect();
    expect(box.left).toBeGreaterThanOrEqual(bounds.left - 1);
    expect(box.right).toBeLessThanOrEqual(bounds.right + 1);
    expect(box.top).toBeGreaterThanOrEqual(bounds.top - 1);
    expect(
      box.bottom,
      element instanceof HTMLElement ? element.outerHTML.slice(0, 180) : 'metadata ending'
    ).toBeLessThanOrEqual(bounds.bottom + 1);
  };

  const expectWrappedRenameHint = async (menu: HTMLElement, dialog: HTMLElement) => {
    await act(() => userEvent.keyboard('{Home}{ArrowDown}'));
    const rename = menu.querySelector<HTMLElement>('[data-menu-item="rename"]')!;
    await vi.waitFor(() => {
      expect(menu.getAttribute('aria-activedescendant')).toBe(rename.id);
      const hint = rename.querySelector('p')!;
      expect(hint.textContent).toBe(TRANSLATIONS['workflowLibrary.renameProjectWorkflowHint']);
      const text = document.createRange();
      text.selectNodeContents(hint);
      expect(text.getClientRects().length).toBeGreaterThan(1);
      expectContained(text, menu);
      expectContained(text, dialog);
    });
  };

  beforeEach(async () => {
    await resizeViewport(1280, 800);
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    onOpenChange = vi.fn((_isOpen: boolean) => {});
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    browse.ensureWorkflowLibraryBrowseLoaded.mockClear();
    browse.loadWorkflowLibraryPlaceTotals.mockClear();
    browse.setWorkflowLibraryBrowseFilter.mockClear();
    loader.open.mockClear();
    loader.replace.mockClear();
    loader.resume.mockClear();
    queryClient.clear();
    templateRecords.clear();
    loader.phase.current = 'idle';
    for (const command of Object.values(COMMANDS)) {
      command.mockClear();
    }
    for (const write of Object.values(libraryWrites)) {
      write.mockClear();
    }
    project.setSnapshot(projectSnapshot(PROJECT_WORKFLOWS));
    preferences.setSnapshot(LIBRARY_PREFERENCES as unknown as WorkflowPreferences);
    setLibraryPreferences.mockClear();
    // The dialog's tab and selection live in the session store; start every test from the account's clean slate.
    accountLifecycle.invalidate();
    // These are the template tabs' behaviours; the project view has its own suite below.
    setWorkflowLibraryTab('user');
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('renders one card per entry with its base model, readiness and leading fact', async () => {
    await openWith(withSnapshot({ entries: [...LOADED_SNAPSHOT.entries, PREVIEW_FIXTURE] }));

    expect(cards()).toHaveLength(5);

    // The project holds a copy of Portrait, so that is its fact; the rail still counts its nodes.
    const portrait = card('wf-portrait')!;
    expect(portrait.textContent).toContain('Portrait Studio');
    expect(portrait.textContent).toContain('SDXL');
    expect(portrait.textContent).toContain('In project');
    expect(portrait.querySelector('[data-workflow-in-project]')).not.toBeNull();
    expect(portrait.querySelector('[data-workflow-readiness]')?.getAttribute('data-workflow-readiness')).toBe('ready');
    expect(document.querySelector('[data-workflow-detail]')?.textContent).toContain('12 nodes');

    const landscape = card('wf-landscape')!;
    expect(landscape.textContent).toContain('FLUX');
    expect(landscape.textContent).toContain('1 model to install');
    expect(landscape.querySelector('[data-workflow-readiness]')?.getAttribute('data-workflow-readiness')).toBe(
      'models'
    );
    expect(landscape.querySelector('[data-workflow-in-project]')).toBeNull();

    const ready = card('wf-preview-fixture')!;
    expect(ready.textContent).toContain('1 node');
    expect(ready.querySelector('[data-workflow-readiness]')?.getAttribute('data-workflow-readiness')).toBe('ready');
  });

  it('marks a workflow whose node types the server lacks with the missing pack count, naming them in the rail', async () => {
    const unknown = (type: string) =>
      buildInvocationNode({ ...PREVIEW_NODE_TEMPLATE, nodePack: 'face-tools', title: type, type }, { x: 400, y: 0 });
    const document_: ProjectGraphState = {
      ...PREVIEW_DOCUMENT,
      nodes: [...PREVIEW_DOCUMENT.nodes, unknown('face_mask'), unknown('face_crop')],
    };
    const custom = entry('wf-custom', 'Custom Nodes', readyEnrichment(3, null, NO_REQUIREMENTS, document_));
    await openWith(withSnapshot({ entries: [custom, LANDSCAPE] }));

    const tile = card('wf-custom')!;
    expect(tile.querySelector('[data-workflow-readiness]')?.getAttribute('data-workflow-readiness')).toBe('nodePacks');
    expect(tile.textContent).toContain('1 node pack missing');
    expect(tile.textContent).not.toContain('3 nodes');
    expect(document.querySelector('[data-workflow-detail] [data-workflow-missing-node-packs]')?.textContent).toBe(
      '1 node pack missing'
    );
    expect(document.querySelector('[data-workflow-detail] [data-workflow-missing-node-pack-names]')?.textContent).toBe(
      'Missing node pack: face-tools'
    );
  });

  it('shows a quiet placeholder while enrichment is pending and omits the badges when it failed', async () => {
    await openWith(LOADED_SNAPSHOT);

    const pending = card('wf-sketch');
    expect(pending?.textContent).toContain('Sketch To Render');
    expect(pending?.textContent).not.toContain('nodes');
    expect(pending?.querySelector('[data-enrichment-placeholder]')).not.toBeNull();

    // A failed enrichment is a quiet omission, never an error-styled card.
    const failed = card('wf-upscale');
    expect(failed?.textContent).toContain('Upscale Chain');
    expect(failed?.textContent).not.toContain('nodes');
    expect(failed?.querySelector('[data-enrichment-placeholder]')).toBeNull();
  });

  it('renders the thumbnail when present, else the node silhouette, else the plain placeholder', async () => {
    await openWith(withSnapshot({ entries: [...LOADED_SNAPSHOT.entries, PREVIEW_FIXTURE] }));

    expect(card('wf-portrait')?.querySelector('img')).not.toBeNull();
    expect(card('wf-portrait')?.querySelector('[data-workflow-silhouette]')).toBeNull();
    // One node, one faint box drawn into a single background image, hidden from assistive technology.
    const silhouette = card('wf-preview-fixture')?.querySelector<HTMLElement>('[data-workflow-silhouette]');
    expect(silhouette?.getAttribute('aria-hidden')).toBe('true');
    expect(silhouette?.dataset.workflowSilhouette).toBe('1');
    expect(silhouette?.childElementCount).toBe(0);
    expect(silhouette?.style.backgroundImage).toContain('data:image/svg+xml');
    // No nodes, or no document yet: the plain no-image placeholder.
    for (const workflowId of ['wf-landscape', 'wf-sketch']) {
      expect(card(workflowId)?.querySelector('img')).toBeNull();
      expect(card(workflowId)?.querySelector('[data-workflow-silhouette]')).toBeNull();
      expect(card(workflowId)?.querySelector('svg.lucide-image-off')).not.toBeNull();
    }
  });

  it('kicks the first load when it opens and leaves the store alone while closed', async () => {
    await act(() => browse.setSnapshot?.(LOADED_SNAPSHOT));
    await renderDialog(false);

    expect(browse.ensureWorkflowLibraryBrowseLoaded).not.toHaveBeenCalled();

    await renderDialog(true);

    // Times are not asserted: StrictMode double-invokes mount effects, and the
    // store's own single-flight is what makes that a no-op.
    expect(browse.ensureWorkflowLibraryBrowseLoaded).toHaveBeenCalled();
  });

  // The toolbar's sort select; the narrow layout's places select is another combobox.
  const sortTrigger = () => document.querySelector<HTMLButtonElement>('[data-workflow-library-sort]');
  /** Opens the Sort menu and returns its radio items: the keys, then the two directions. */
  const openSortMenu = async () => {
    const trigger = sortTrigger()!;
    await act(() => userEvent.click(trigger));
    await vi.waitFor(() => expect(trigger.getAttribute('aria-expanded')).toBe('true'));
    const menu = document.getElementById(trigger.getAttribute('aria-controls') ?? '')!;
    return [...menu.querySelectorAll<HTMLElement>('[role="menuitemradio"]')];
  };
  const chooseSortItem = async (label: string) => {
    const item = (await openSortMenu()).find((candidate) => candidate.textContent === label);
    expect(item, label).not.toBeUndefined();
    await act(() => userEvent.click(item!));
  };

  it('opens in the account’s stored order and refetches in the key chosen, starting in its natural direction', async () => {
    preferences.setSnapshot({
      ...LIBRARY_PREFERENCES,
      workflowLibrarySort: 'opened',
    } as unknown as WorkflowPreferences);
    await openWith(withSnapshot({ filter: { ...LOADED_SNAPSHOT.filter, sort: 'opened' } }));

    expect(browse.ensureWorkflowLibraryBrowseLoaded).toHaveBeenCalledWith({
      direction: 'desc',
      place: 'user',
      search: '',
      sort: 'opened',
      tags: [],
    });
    // My library's own first page counts it; the probes cover the other places.
    expect(browse.loadWorkflowLibraryPlaceTotals).toHaveBeenCalledWith(['recent', 'default']);
    // The control shows the key; its name spells out the key and the direction its glyph shows.
    expect(sortTrigger()?.textContent).toBe('Last opened');
    expect(sortTrigger()?.getAttribute('aria-label')).toBe('Sort: Last opened, Newest first');
    const items = await openSortMenu();
    expect(items.map((item) => item.textContent)).toEqual([
      'Last opened',
      'Last updated',
      'Last run',
      'Date created',
      'Name',
      'Newest first',
      'Oldest first',
    ]);
    await act(() => userEvent.click(items.find((item) => item.textContent === 'Name')!));

    expect(setLibraryPreferences).toHaveBeenCalledWith({
      workflowLibrarySort: 'name',
      workflowLibrarySortDirection: 'asc',
    });
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ direction: 'asc', sort: 'name' });
  });

  it('reverses My library’s order from the Sort menu and keeps the direction with the account', async () => {
    await openWith(LOADED_SNAPSHOT);

    await chooseSortItem('Oldest first');

    expect(setLibraryPreferences).toHaveBeenCalledWith({ workflowLibrarySortDirection: 'asc' });
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ direction: 'asc' });
  });

  it('lists Templates by name only, reversing them for the visit without touching the account’s order', async () => {
    setWorkflowLibraryTab('default');
    await openWith(withSnapshot({ filter: { ...LOADED_SNAPSHOT.filter, place: 'default' } }));

    expect(sortTrigger()?.getAttribute('aria-label')).toBe('Sort: Name, A to Z');
    const items = await openSortMenu();
    // Names list A to Z first; the only key is fixed.
    expect(items.map((item) => item.textContent)).toEqual(['Name', 'A to Z', 'Z to A']);
    expect(items[0]?.hasAttribute('data-disabled')).toBe(true);
    await act(() => userEvent.click(items[2]!));

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ direction: 'desc', sort: 'name' });
    expect(setLibraryPreferences).not.toHaveBeenCalled();
  });

  it('opens Recent in its fixed order, most recently opened first', async () => {
    await openWith(LOADED_SNAPSHOT);

    await clickPlace('recent');

    expect(workflowUiStore.getSnapshot().libraryTab).toBe('recent');
    expect(placeButton('recent')?.getAttribute('aria-current')).toBe('page');
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({
      direction: 'desc',
      place: 'recent',
      search: '',
      sort: 'updated',
      tags: [],
    });
    expect(sortTrigger()?.textContent).toBe('Last opened');
    expect(sortTrigger()?.disabled).toBe(true);
  });

  it('keeps the toolbar one row of equal-height controls 8px apart, New workflow last', async () => {
    await resizeViewport(1600, 1000);
    setWorkflowLibraryTab('project');
    await openWith(LOADED_SNAPSHOT);
    const toolbar = document.querySelector<HTMLElement>('[data-workflow-library-toolbar]')!;
    const shown = [...toolbar.children].filter((child) => isShown(child as HTMLElement)) as HTMLElement[];
    const boxes = shown.map((child) => child.getBoundingClientRect());

    expect(shown.map((child) => child.textContent)).toEqual(['', 'Project order', '', 'New workflow']);
    // Same-named control sizes share one outer height.
    expect(new Set(boxes.map((box) => Math.round(box.height))).size).toBe(1);
    expect(new Set(boxes.map((box) => Math.round(box.top)))).toEqual(new Set([Math.round(boxes[0]!.top)]));
    for (let index = 1; index < boxes.length; index += 1) {
      expect(Math.round(boxes[index]!.left - boxes[index - 1]!.right)).toBe(8);
    }
  });

  it('fills thumbnails by default and letterboxes them once Fit is chosen from the view options', async () => {
    await openWith(LOADED_SNAPSHOT);
    const thumbnail = () => card('wf-portrait')!.querySelector('img')!;
    const fitItem = () => document.querySelector<HTMLElement>('[role="menuitemradio"][data-value="contain"]');

    expect(getComputedStyle(thumbnail()).objectFit).toBe('cover');

    await act(() => userEvent.click(document.querySelector<HTMLElement>('button[aria-label="View options"]')!));
    await vi.waitFor(() => expect(fitItem()).not.toBeNull());
    expect(fitItem()?.getAttribute('aria-checked')).toBe('false');
    await act(() => userEvent.click(fitItem()!));

    expect(setLibraryPreferences).toHaveBeenCalledWith({ workflowLibraryThumbnailFit: 'contain' });
    expect(getComputedStyle(thumbnail()).objectFit).toBe('contain');
  });

  it('switches cards to the large size from the view options and keeps the choice with the account', async () => {
    await openWith(LOADED_SNAPSHOT);
    const sizeItem = (value: string) =>
      document.querySelector<HTMLElement>(`[role="menuitemradio"][data-value="${value}"]`);

    await act(() => userEvent.click(document.querySelector<HTMLElement>('button[aria-label="View options"]')!));
    await vi.waitFor(() => expect(sizeItem('large')).not.toBeNull());
    expect(sizeItem('small')?.getAttribute('aria-checked')).toBe('true');
    await act(() => userEvent.click(sizeItem('large')!));

    expect(setLibraryPreferences).toHaveBeenCalledWith({ workflowLibraryCardSize: 'large' });
    expect(sizeItem('large')?.getAttribute('aria-checked')).toBe('true');
  });

  it('switches to the bundled defaults once when the account has no workflows of its own', async () => {
    await openWith(withSnapshot({ placeTotals: { user: 0 } }));
    await wait(0);

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ place: 'default', tags: [] });
    // The rail follows the store's place.
    expect(workflowUiStore.getSnapshot().libraryTab).toBe('default');
    expect(placeButton('default')?.getAttribute('aria-current')).toBe('page');
  });

  it('keeps Shared selected when the owner has no workflows of their own', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    setWorkflowLibraryTab('shared');
    await openWith(
      withSnapshot({
        filter: { direction: 'desc', place: 'shared', search: '', sort: 'updated', tags: [] },
        placeTotals: { user: 0 },
      })
    );
    await wait(0);

    expect(browse.setWorkflowLibraryBrowseFilter).not.toHaveBeenCalledWith(
      expect.objectContaining({ place: 'default' })
    );
    expect(workflowUiStore.getSnapshot().libraryTab).toBe('shared');
  });

  it('leaves the place alone when the account already has workflows', async () => {
    await openWith(LOADED_SNAPSHOT);
    await wait(0);

    expect(browse.setWorkflowLibraryBrowseFilter).not.toHaveBeenCalledWith(
      expect.objectContaining({ place: 'default' })
    );
  });

  it('switches to Templates only for the place the library opened on', async () => {
    setWorkflowLibraryTab('project');
    await openWith(withSnapshot({ placeTotals: { user: 0 } }));
    // Opening on This project probes every library place, once.
    expect(browse.loadWorkflowLibraryPlaceTotals).toHaveBeenCalledWith(['recent', 'user', 'default']);

    await clickPlace('user');
    await wait(0);
    await clickPlace('project');
    await clickPlace('user');
    await wait(0);

    // An empty My library chosen in the rail stays chosen, and leaving This project probes nothing again.
    expect(workflowUiStore.getSnapshot().libraryTab).toBe('user');
    expect(browse.setWorkflowLibraryBrowseFilter).not.toHaveBeenCalledWith(
      expect.objectContaining({ place: 'default' })
    );
    expect(new Set(browse.loadWorkflowLibraryPlaceTotals.mock.calls.map(([places]) => places.join()))).toEqual(
      new Set(['recent,user,default'])
    );
  });

  it('clears the checked tags but keeps the search when the place changes', async () => {
    await openWith(LOADED_SNAPSHOT);
    const input = document.querySelector<HTMLInputElement>('input[type="search"]')!;
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    await act(async () => {
      setValue?.call(input, 'portrait');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    await clickTag('portrait');
    browse.setWorkflowLibraryBrowseFilter.mockClear();

    // Before the search's debounce: the place change carries the typed search, not the store's.
    await clickPlace('default');

    // The store applies filter patches literally, so clearing the tags on a place switch is the dialog's job.
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({
      direction: 'desc',
      place: 'default',
      search: 'portrait',
      sort: 'updated',
      tags: [],
    });
    expect(input.value).toBe('portrait');
    await wait(400);
    // The cancelled debounce never lands on the new place.
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledTimes(1);
    expect(workflowUiStore.getSnapshot().libraryQuery.tags).toEqual([]);
  });

  it('shows place totals in the rail and opens Shared with the public filter', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    await openWith(withSnapshot({ placeTotals: { default: 25, recent: 3, shared: 9, user: 4 } }));

    expect(
      [...document.querySelectorAll<HTMLElement>('[data-library-place]')].map((place) => place.textContent)
    ).toEqual(['In this project3', 'Recent3', 'My library4', 'Templates25', 'Shared with me9']);
    expect(browse.loadWorkflowLibraryPlaceTotals).toHaveBeenCalledWith(['recent', 'default', 'shared']);

    await clickPlace('shared');

    expect(workflowUiStore.getSnapshot().libraryTab).toBe('shared');
    expect(placeButton('shared')?.getAttribute('aria-current')).toBe('page');
    expect(placeButton('user')?.hasAttribute('aria-current')).toBe(false);
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith(expect.objectContaining({ place: 'shared' }));
    // The centre column is named by the place it shows.
    expect(document.querySelector('section[aria-labelledby]')?.getAttribute('aria-labelledby')).toBe(
      placeButton('shared')?.querySelector('[id]')?.id
    );
  });

  it('hides Shared with me when multiuser mode is disabled', async () => {
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    await openWith(LOADED_SNAPSHOT);

    expect(placeButton('shared')).toBeNull();
  });

  it('hides template editing controls for another user’s Shared workflow', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    setWorkflowLibraryTab('shared');
    await openWith(
      withSnapshot({
        entries: [NON_OWNER_SHARED],
        filter: { direction: 'desc', place: 'shared', search: '', sort: 'updated', tags: [] },
      })
    );

    const moreActions = document.querySelector<HTMLButtonElement>('[data-workflow-detail] [aria-label="More actions"]');
    expect(moreActions).not.toBeNull();
    await act(async () => {
      moreActions?.click();
      await settleFrame();
    });
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).not.toBeNull());

    expect(document.querySelector('[data-menu-item="rename"]')).toBeNull();
    expect(document.querySelector('[data-menu-item="delete"]')).toBeNull();
    expect(document.querySelector('[aria-label="workflowLibrary.thumbnailChoose"]')).toBeNull();
    expect(document.querySelector('[data-thumbnail-remove]')).toBeNull();
  });

  it('keeps an owner’s Shared workflow editable but hides its visibility switch', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    setWorkflowLibraryTab('shared');
    await openWith(
      withSnapshot({
        entries: [{ ...PORTRAIT, item: { ...PORTRAIT.item, user_id: 'user-a' } }],
        filter: { direction: 'desc', place: 'shared', search: '', sort: 'updated', tags: [] },
      })
    );

    const moreActions = document.querySelector<HTMLButtonElement>('[data-workflow-detail] [aria-label="More actions"]');
    expect(moreActions).not.toBeNull();
    await act(async () => {
      moreActions?.click();
      await settleFrame();
    });
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).not.toBeNull());

    expect(document.querySelector('[data-menu-item="rename"]')).not.toBeNull();
    expect(document.querySelector('[data-menu-item="delete"]')).not.toBeNull();
    expect(document.querySelector('[aria-label="workflowLibrary.thumbnailChoose"]')).not.toBeNull();
    expect(document.querySelector('[data-workflow-detail] input[type="checkbox"]')).toBeNull();
  });

  it('shows the visibility state on saved user cards and offers the switch only to the owner in Yours', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    setWorkflowLibraryTab('user');
    await openWith(
      withSnapshot({
        entries: [OWNER_PRIVATE],
        filter: { direction: 'desc', place: 'user', search: '', sort: 'updated', tags: [] },
      })
    );

    // Only public templates carry a visibility badge; the rail states it either way.
    expect(card('wf-portrait')?.textContent).not.toContain('Private');
    expect(card('wf-portrait')?.querySelector('[data-workflow-visibility]')).toBeNull();
    expect(document.querySelector('[data-workflow-detail] [data-workflow-visibility]')?.textContent).toBe('Private');
    const ownerSwitch = document.querySelector<HTMLInputElement>('[data-workflow-detail] input[type="checkbox"]');
    expect(ownerSwitch).not.toBeNull();
    expect(ownerSwitch?.checked).toBe(false);
    expect(ownerSwitch?.labels?.[0]?.textContent).toBe('Public workflow');

    await renderDialog();
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-b' } };
    await openWith(
      withSnapshot({
        entries: [{ ...OWNER_PRIVATE, item: { ...OWNER_PRIVATE.item, user_id: 'user-a' } }],
        filter: { direction: 'desc', place: 'user', search: '', sort: 'updated', tags: [] },
      })
    );

    expect(document.querySelector('[data-workflow-detail] input[type="checkbox"]')).toBeNull();
    const moreActions = document.querySelector<HTMLButtonElement>('[data-workflow-detail] [aria-label="More actions"]');
    await act(async () => {
      moreActions?.click();
      await settleFrame();
    });
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).not.toBeNull());
    expect(document.querySelector('[data-menu-item="rename"]')).toBeNull();
    expect(document.querySelector('[data-menu-item="delete"]')).toBeNull();
    expect(document.querySelector('[aria-label="workflowLibrary.thumbnailChoose"]')).toBeNull();
  });

  it('shows public state on Shared cards without visibility controls', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    setWorkflowLibraryTab('shared');
    await openWith(
      withSnapshot({
        entries: [OWNER_PUBLIC],
        filter: { direction: 'desc', place: 'shared', search: '', sort: 'updated', tags: [] },
      })
    );

    expect(card('wf-portrait')?.querySelector('[data-workflow-visibility]')?.textContent).toBe('Public');
    expect(document.querySelector('[data-workflow-detail] input[type="checkbox"]')).toBeNull();
  });

  it('keeps visibility out of single-user cards and rails, even for a public template', async () => {
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    await openWith(withSnapshot({ entries: [OWNER_PUBLIC] }));

    expect(document.querySelector('[data-workflow-visibility]')).toBeNull();
    expect(document.querySelector('[data-workflow-detail] input[type="checkbox"]')).toBeNull();
    expect(document.querySelector('[data-workflow-detail]')?.textContent).not.toContain('Visibility');
  });

  it('keeps Yours template editing available in single-user mode', async () => {
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    setWorkflowLibraryTab('user');
    await openWith(
      withSnapshot({
        entries: [PORTRAIT],
        filter: { direction: 'desc', place: 'user', search: '', sort: 'updated', tags: [] },
      })
    );

    const moreActions = document.querySelector<HTMLButtonElement>('[data-workflow-detail] [aria-label="More actions"]');
    expect(moreActions).not.toBeNull();
    await act(async () => {
      moreActions?.click();
      await settleFrame();
    });
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).not.toBeNull());

    expect(document.querySelector('[data-menu-item="rename"]')).not.toBeNull();
    expect(document.querySelector('[data-menu-item="delete"]')).not.toBeNull();
    expect(document.querySelector('[aria-label="workflowLibrary.thumbnailChoose"]')).not.toBeNull();
    expect(document.querySelector('[data-workflow-detail] input[type="checkbox"]')).toBeNull();
  });

  it('asks for any of the checked tags, names them in the result line, and clears them', async () => {
    await openWith(LOADED_SNAPSHOT);
    // StrictMode's remount of the browse session clears the listing's query once on mount.
    browse.setWorkflowLibraryBrowseFilter.mockClear();
    const results = () => document.querySelector('[data-workflow-library-results]')?.textContent;

    expect(results()).toBe('24 workflows');
    expect(document.body.textContent).toContain('Showing workflows with any checked tag');

    await clickTag('portrait');
    await clickTag('upscale');

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenNthCalledWith(1, { tags: ['portrait'] });
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenNthCalledWith(2, { tags: ['portrait', 'upscale'] });
    expect(tagCheckbox('portrait')?.checked).toBe(true);
    expect(results()).toBe('24 workflows · portrait or upscale');

    await clickText('Clear');

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenLastCalledWith({ tags: [] });
    expect(tagCheckbox('portrait')?.checked).toBe(false);
    expect(buttonWithText('Clear')).toBeUndefined();
  });

  it('keeps a checked tag the counts stop listing until it is unchecked, then keeps focus in the rail', async () => {
    await openWith(LOADED_SNAPSHOT);
    await clickTag('portrait');
    await act(() => browse.setSnapshot?.(withSnapshot({ tagCounts: [] })));

    expect(tagCheckbox('portrait')?.checked).toBe(true);
    await act(() => tagCheckbox('portrait')!.focus());
    await act(() => userEvent.keyboard(' '));

    expect(tagCheckbox('portrait')).toBeNull();
    expect(document.activeElement).toBe(placeButton('user'));
  });

  it('clears the listing’s search and tags on close, so reopening on a library place loads unfiltered', async () => {
    await act(() => browse.setSnapshot?.(LOADED_SNAPSHOT));
    setWorkflowLibraryOpen(true);
    await act(async () => {
      root.render(
        <StrictMode>
          <QueryClientProvider client={queryClient}>
            <ChakraProvider value={system}>
              <WorkflowUiProvider adapter={UI_ADAPTER}>
                <GalleryHostProvider host={GALLERY_HOST}>
                  <WorkflowGraphPreviewProvider adapter={GRAPH_PREVIEW}>
                    <StoreBoundLibraryDialog />
                  </WorkflowGraphPreviewProvider>
                </GalleryHostProvider>
              </WorkflowUiProvider>
            </ChakraProvider>
          </QueryClientProvider>
        </StrictMode>
      );
      await settleFrame();
    });
    const search = () => document.querySelector<HTMLInputElement>('input[type="search"]');
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    await act(async () => {
      setValue?.call(search(), 'photo');
      search()!.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    await clickTag('portrait');
    await wait(400);
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ search: 'photo' });
    browse.setWorkflowLibraryBrowseFilter.mockClear();

    await act(async () => {
      document.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click();
      await settleFrame();
    });

    // The listing drops the query as the dialog closes, not when it next opens.
    expect(workflowUiStore.getSnapshot().isLibraryOpen).toBe(false);
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenLastCalledWith({ search: '', tags: [] });
    browse.ensureWorkflowLibraryBrowseLoaded.mockClear();

    await act(async () => {
      setWorkflowLibraryOpen(true);
      await settleFrame();
    });

    expect(workflowUiStore.getSnapshot().libraryTab).toBe('user');
    expect(browse.ensureWorkflowLibraryBrowseLoaded).toHaveBeenCalledWith(
      expect.objectContaining({ place: 'user', search: '', tags: [] })
    );
    expect(search()?.value).toBe('');
    expect(tagCheckbox('portrait')?.checked).toBe(false);
  });

  it('pushes the search term to the store once, after the debounce', async () => {
    await openWith(LOADED_SNAPSHOT);

    const input = document.querySelector<HTMLInputElement>('input[type="search"]');
    expect(input).not.toBeNull();

    // React tracks the input's value through its own setter, so assigning
    // `.value` directly would be swallowed as a no-op change.
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    // StrictMode's remount of the browse session clears the listing's query once on mount.
    browse.setWorkflowLibraryBrowseFilter.mockClear();

    for (const value of ['p', 'ph', 'pho', 'phot', 'photo']) {
      await act(async () => {
        setValue?.call(input, value);
        input?.dispatchEvent(new Event('input', { bubbles: true }));
        await Promise.resolve();
      });
    }

    expect(browse.setWorkflowLibraryBrowseFilter).not.toHaveBeenCalled();

    await wait(400);

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledTimes(1);
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ search: 'photo' });
  });

  it('bounds mounted cards and requests server positions while scrolling 1,000 workflows', async () => {
    await resizeViewport(1280, 800);
    const entries = Array.from({ length: 1000 }, (_, index) => ({
      ...LOADED_SNAPSHOT.entries[0]!,
      item: { ...LOADED_SNAPSHOT.entries[0]!.item, workflow_id: `large-${index}`, name: `Large ${index}` },
    }));
    await openWith({ ...LOADED_SNAPSHOT, entries, total: 1000, pages: 50 });
    expect(cards().length).toBeGreaterThan(0);
    expect(cards().length).toBeLessThan(40);
    const viewport = gridViewport()!;
    await act(async () => {
      viewport.scrollTop = 3200;
      viewport.dispatchEvent(new Event('scroll'));
      await settleFrame();
    });
    expect(cards().length).toBeLessThan(40);
    expect(browse.loadWorkflowLibraryRange.mock.calls.some(([start]) => Number(start) > 0)).toBe(true);
    await act(async () => {
      viewport.scrollTop = 0;
      viewport.dispatchEvent(new Event('scroll'));
      await settleFrame();
    });
    expect(card('large-0')).not.toBeNull();
  });

  it('keeps measured rows separated after viewport resize, zoom and text reflow', async () => {
    const entries = Array.from({ length: 90 }, (_, index) => ({
      ...PORTRAIT,
      item: { ...PORTRAIT.item, workflow_id: `reflow-${index}`, name: LONG_NAME },
    }));
    await openWith({ ...LOADED_SNAPSHOT, entries, total: 90, pages: 5 });
    const rootStyle = document.documentElement.style;
    const previousZoom = rootStyle.zoom;
    const previousFontSize = rootStyle.fontSize;
    onTestFinished(() => {
      rootStyle.zoom = previousZoom;
      rootStyle.fontSize = previousFontSize;
    });

    const expectRowGaps = async (zoom = 1, gap = 12) => {
      await vi.waitFor(() => {
        const rows = [...new Set(cards().map((element) => element.closest<HTMLElement>('[data-index]')!))].sort(
          (a, b) => Number(a.dataset.index) - Number(b.dataset.index)
        );
        expect(rows.length).toBeGreaterThanOrEqual(3);
        expect(cards().length).toBeLessThan(40);
        for (let index = 1; index < rows.length; index += 1) {
          const previous = rows[index - 1]!;
          const current = rows[index]!;
          const previousRect = previous.getBoundingClientRect();
          const currentRect = current.getBoundingClientRect();
          // Row padding owns the 12px gap; the row boxes themselves must meet without overlap.
          expect(Math.abs((currentRect.top - previousRect.bottom) / zoom)).toBeLessThanOrEqual(1);
          const cardBottom = Math.max(
            ...[...previous.querySelectorAll<HTMLElement>('[data-workflow-card]')].map(
              (element) => element.getBoundingClientRect().bottom
            )
          );
          expect(Math.abs((currentRect.top - cardBottom) / zoom - gap)).toBeLessThanOrEqual(1);
        }
      });
    };

    await expectRowGaps();
    await page.viewport(700, 800);
    await expectRowGaps();
    await page.viewport(1280, 800);
    await expectRowGaps();
    await act(() => {
      rootStyle.zoom = '2';
    });
    await expectRowGaps(2);
    await act(() => {
      rootStyle.zoom = previousZoom;
      rootStyle.fontSize = '20px';
    });
    // Text scaling also scales the rem-based gap.
    await expectRowGaps(1, 15);
    await act(() => {
      const viewport = gridViewport()!;
      viewport.scrollTop = 1000;
      viewport.dispatchEvent(new Event('scroll'));
    });
    await page.viewport(900, 800);
    await expectRowGaps(1, 15);
  });

  it.each(
    (['library', 'project'] as const).flatMap((view) =>
      [
        [640, 400],
        [512, 384],
        [320, 400],
        [1280, 320],
        [700, 320],
        [640, 320],
        [512, 320],
        [320, 320],
      ].map(([width, height]) => ({ view, width: width!, height: height! }))
    )
  )(
    'keeps the $view grid, selected metadata and actions usable at $width×$height layout viewport',
    async ({ view, width: targetWidth, height: targetHeight }) => {
      const description = `${'Required workflow metadata. '.repeat(200)}End of required metadata.`;
      const entries = [
        { ...PREVIEW_FIXTURE, item: { ...PREVIEW_FIXTURE.item, description } },
        ...Array.from({ length: 1000 }, (_, index) => ({
          ...PORTRAIT,
          item: { ...PORTRAIT.item, workflow_id: `layout-${index}` },
        })),
      ];
      project.setSnapshot(
        projectSnapshot([
          ...PROJECT_WORKFLOWS.map((entry) =>
            entry === USER_COPY ? { ...entry, document: { ...entry.document, description } } : entry
          ),
          ...Array.from({ length: 1000 }, (_, index) => ({
            ...USER_COPY,
            document: { ...USER_COPY.document, id: `project-layout-${index}` },
          })),
        ])
      );
      await openWith(withSnapshot({ entries, total: entries.length, pages: 51 }));
      if (view === 'project') {
        await clickPlace('project');
        await vi.waitFor(() => expect(card('wf-user')).not.toBeNull());
        await act(() => card('wf-user')!.click());
        expect(document.querySelector('[data-project-workflow-detail="wf-user"]')).not.toBeNull();
      }

      const dialog = document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]')!;
      const detail = dialog.querySelector<HTMLElement>('[data-workflow-detail], [data-project-workflow-detail]')!;
      const metadata = detail.querySelector<HTMLElement>('[role="region"]')!;
      const summary = detail.querySelector<HTMLElement>('[data-detail-sheet-summary]')!;
      const toggle = summary.querySelector<HTMLButtonElement>('[data-detail-sheet-toggle]')!;
      const grid = gridViewport()!;
      const close = dialog.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!;
      const preview = [...detail.querySelectorAll<HTMLButtonElement>('button')].find(
        (button) => button.textContent === 'Preview graph'
      )!;
      // The wide rail carries Open and More itself; the narrow layout's sheet summary carries them instead.
      const railActions = () => {
        const owner = isNarrow() ? summary : metadata.closest<HTMLElement>('[data-detail-sheet-details]')!;
        return {
          menu: owner.querySelector<HTMLButtonElement>('button[aria-label="More actions"]')!,
          open: [...owner.querySelectorAll<HTMLButtonElement>('button')].find(
            (button) => button.textContent === 'Open'
          )!,
        };
      };
      const isNarrow = () => dialog.getBoundingClientRect().width < 56 * 16;

      for (const [width, height] of [
        [1280, 800],
        [700, 800],
        [targetWidth, targetHeight],
      ] as const) {
        // Real layout-viewport reduction, as at 200% browser zoom on a 1280×800 display. CSS zoom misses this.
        await page.viewport(width, height);
        expect(window.innerWidth).toBe(width);
        expect(window.innerHeight).toBe(height);
        await vi.waitFor(() => {
          const narrow = isNarrow();
          // Enough for a substantial part of a card, rather than merely nonzero boxes.
          expect(grid.clientHeight, 'workflow grid scroll viewport').toBeGreaterThanOrEqual(120);
          expect(grid.clientHeight, 'finite virtual grid viewport').toBeLessThanOrEqual(640);
          if (height <= 400) {
            expect(grid.clientHeight, 'flow grid has a finite viewport').toBe(256);
          }
          if (narrow) {
            // The sheet's summary stays in view under the grid; its details wait behind the disclosure.
            expect(isShown(summary), 'sheet summary').toBe(true);
            expect(isShown(metadata), 'collapsed sheet details').toBe(false);
            expectContained(summary, dialog);
          } else {
            expect(isShown(summary), 'wide layout has no sheet').toBe(false);
            // Enough for readable metadata.
            expect(metadata.clientHeight, 'selected metadata scroll viewport').toBeGreaterThanOrEqual(80);
            expect(detail.clientHeight, 'finite detail pane').toBeLessThanOrEqual(640);
            if (height <= 400) {
              expect(detail.getBoundingClientRect().height, 'flow detail does not grow with metadata').toBe(272);
            }
            expect(grid.getBoundingClientRect().right).toBeLessThanOrEqual(detail.getBoundingClientRect().left);
          }
          expect(dialog.scrollWidth).toBeLessThanOrEqual(dialog.clientWidth + 1);
          expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(width);
          expect(dialog.getBoundingClientRect().top).toBeGreaterThanOrEqual(0);
          expect(dialog.getBoundingClientRect().bottom).toBeLessThanOrEqual(height);
          // Short windows scroll the centre column to the rail's actions; the focus walk below proves they stay
          // reachable and unclipped.
          if (height >= 640) {
            const { menu, open } = railActions();
            for (const action of narrow ? [open, menu, toggle, close] : [open, preview, menu, close]) {
              expectContained(action, dialog);
            }
          }
          for (const control of dialog.querySelectorAll<HTMLElement>(
            'input[type="search"], [role="combobox"], [data-library-place], [data-workflow-library-sort], [data-workflow-library-narrow-options]'
          )) {
            if (isShown(control)) {
              expectContained(control, dialog);
            }
          }
          expect(grid.scrollWidth).toBeLessThanOrEqual(grid.clientWidth + 1);
          expect(cards().length).toBeGreaterThan(0);
          expect(cards().length).toBeLessThan(40);
          for (const element of cards()) {
            const box = element.getBoundingClientRect();
            const bounds = grid.getBoundingClientRect();
            expect(box.left).toBeGreaterThanOrEqual(bounds.left);
            expect(box.right).toBeLessThanOrEqual(bounds.right + 1);
            expect(box.width).toBeGreaterThanOrEqual(150);
            const row = element.closest<HTMLElement>('[data-index]')!;
            expectContained(element, row);
            const nextRow = row.nextElementSibling;
            if (nextRow) {
              expect(box.bottom).toBeLessThanOrEqual(nextRow.getBoundingClientRect().top);
            }
          }
        });
      }

      const expectUnclipped = (element: HTMLElement) => {
        expectContained(element, dialog);
        for (let parent = element.parentElement; parent && parent !== dialog; parent = parent.parentElement) {
          const style = getComputedStyle(parent);
          // A box-less ancestor (`display: contents`) clips nothing.
          if (style.display !== 'contents' && /(auto|scroll|hidden|clip)/.test(style.overflowY)) {
            expectContained(element, parent);
          }
        }
      };
      await act(() => cards()[0]!.focus());
      const focusedCard = cards()[0]!.getBoundingClientRect();
      const gridBounds = grid.getBoundingClientRect();
      const modalBounds = dialog.getBoundingClientRect();
      expect(
        Math.min(focusedCard.bottom, gridBounds.bottom, modalBounds.bottom) -
          Math.max(focusedCard.top, gridBounds.top, modalBounds.top)
      ).toBeGreaterThanOrEqual(120);
      await act(() => grid.focus());
      await act(() => userEvent.keyboard('{PageDown}'));
      await vi.waitFor(() => {
        expect(grid.scrollTop).toBeGreaterThan(0);
        expect(cards().length).toBeLessThan(40);
        expect(grid.clientHeight).toBeLessThanOrEqual(640);
      });
      if (isNarrow()) {
        // The full rail opens over the grid from the sheet's disclosure, focus moving into it.
        await act(() => toggle.focus());
        await act(() => userEvent.keyboard('{Enter}'));
        expect(toggle.getAttribute('aria-expanded')).toBe('true');
        expect(isShown(metadata)).toBe(true);
        expect(metadata.clientHeight, 'expanded sheet metadata scroll viewport').toBeGreaterThanOrEqual(80);
      }
      const { menu, open } = railActions();
      await act(() => metadata.focus());
      expect(document.activeElement).toBe(metadata);
      expectUnclipped(metadata);
      await act(() => userEvent.keyboard('{End}'));
      await vi.waitFor(() => {
        expect(metadata.scrollTop).toBeGreaterThan(0);
        // The facts list closes both rails.
        const lastMetadata = [...metadata.querySelectorAll<HTMLElement>('dd')].at(-1)!;
        const ending = document.createRange();
        ending.selectNodeContents(lastMetadata);
        const text = lastMetadata.firstChild!;
        ending.setStart(text, Math.max(0, text.textContent!.length - 24));
        expectContained(ending, metadata);
        expectContained(ending, dialog);
      });

      // Native focus scrolling must bring each action into the modal's clipped viewport, including flow layouts.
      const reached = new Set<Element>();
      for (let index = 0; index < 40 && reached.size < 4; index += 1) {
        await act(() => userEvent.keyboard('{Tab}'));
        if ([open, preview, menu, close].includes(document.activeElement as HTMLButtonElement)) {
          expectUnclipped(document.activeElement as HTMLElement);
          reached.add(document.activeElement!);
        }
      }
      expect(reached.size).toBe(4);
      const toolbarTargets = [
        dialog.querySelector<HTMLElement>('input[type="search"]'),
        [...dialog.querySelectorAll<HTMLElement>('[role="combobox"]')].find(isShown),
        [...dialog.querySelectorAll<HTMLElement>('[data-library-place][aria-current="page"]')].find(isShown),
        [...dialog.querySelectorAll<HTMLButtonElement>('button')].find(
          (button) => button.textContent === 'New workflow'
        ),
      ].filter((element): element is HTMLElement => Boolean(element));
      const toolbarReached = new Set<Element>();
      for (const key of ['{Shift>}{Tab}{/Shift}', '{Tab}']) {
        await act(() => close.focus());
        for (let index = 0; index < 30; index += 1) {
          await act(() => userEvent.keyboard(key));
          const focused = document.activeElement as HTMLElement;
          if (focused.hasAttribute('data-workflow-card')) {
            break;
          }
          if (toolbarTargets.includes(focused)) {
            expectUnclipped(focused);
            toolbarReached.add(focused);
          }
        }
      }
      expect(toolbarReached.size).toBe(toolbarTargets.length);
      if (dialog.scrollHeight > dialog.clientHeight + 1) {
        await act(() => close.focus());
        await act(() => userEvent.keyboard('{End}'));
        await vi.waitFor(() => {
          expect(dialog.scrollTop).toBeGreaterThan(0);
          expectUnclipped(preview);
        });
      }
      // Focus first, as a keyboard user would, so offscreen controls scroll into view before activation.
      await act(() => menu.focus());
      expectUnclipped(menu);
      await act(() => userEvent.keyboard('{Enter}'));
      await vi.waitFor(() => expect(menu.getAttribute('aria-expanded')).toBe('true'));
      const menuContent = document.querySelector<HTMLElement>('[role="menu"]')!;
      // Open state precedes the entry slide's final geometry.
      await act(() => settleAnimations(menuContent));
      expectContained(menuContent, dialog);
      if (view === 'project' && targetWidth === 320) {
        await expectWrappedRenameHint(menuContent, dialog);
      }
      await act(() => userEvent.keyboard('{End}'));
      const lastAction = [...menuContent.querySelectorAll<HTMLElement>('[role="menuitem"]')].at(-1)!;
      await vi.waitFor(() => {
        expect(menuContent.getAttribute('aria-activedescendant')).toBe(lastAction.id);
        expectContained(lastAction, menuContent);
        expectContained(lastAction, dialog);
      });
      await act(() => userEvent.keyboard('{Escape}'));
      await act(() => preview.focus());
      expectUnclipped(preview);
      await act(() => userEvent.keyboard('{Enter}'));
      await waitForPreviewDialog();
      expect(document.querySelector('[data-preview-dialog]')).not.toBeNull();
      // The preview renderer is stubbed; close it through its existing harness controls before using the rail.
      await clickText('Close preview');
      await clickText('Finish preview exit');
      await act(() => open.focus());
      expectUnclipped(open);
      await act(() => userEvent.keyboard('{Enter}'));
      expect(view === 'project' ? COMMANDS.selectWorkflow : loader.open).toHaveBeenCalled();
      await act(() => close.focus());
      expectUnclipped(close);
      await act(() => userEvent.keyboard('{Enter}'));
      expect(onOpenChange).toHaveBeenCalledWith(false);
    }
  );

  it('keeps keyboard focus mounted across virtual row boundaries', async () => {
    await resizeViewport(1280, 800);
    const entries = Array.from({ length: 90 }, (_, index) => ({
      ...PORTRAIT,
      item: { ...PORTRAIT.item, workflow_id: `key-${index}` },
    }));
    await openWith({ ...LOADED_SNAPSHOT, entries, total: 90, pages: 5 });
    await act(() => document.querySelector<HTMLElement>('[data-workflow-card-menu="key-0"]')?.focus());
    for (let index = 1; index <= 18; index += 1) {
      await act(async () => {
        await userEvent.keyboard('{Tab}');
        await settleFrame();
        await settleFrame();
      });
      await vi.waitFor(() => expect(document.activeElement?.getAttribute('data-workflow-card')).toBe(`key-${index}`));
      await act(() => userEvent.keyboard('{Tab}'));
      expect(document.activeElement?.getAttribute('data-workflow-card-menu')).toBe(`key-${index}`);
    }
    expect(cards().length).toBeLessThan(40);
  });

  it.each(['library', 'project'] as const)('keeps the %s card menu inside its dialog while resizing', async (view) => {
    await resizeViewport(320, 320);
    await openWith(LOADED_SNAPSHOT);
    if (view === 'project') {
      await clickPlace('project');
    }
    await vi.waitFor(() => expect(card(view === 'project' ? 'wf-alpha' : 'wf-landscape')).not.toBeNull());
    const source = card(view === 'project' ? 'wf-alpha' : 'wf-landscape')!;
    await act(() => source.focus());
    // Keyboard context-menu events have no pointer coordinates.
    await act(() => source.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true })));
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).not.toBeNull()
    );
    const menu = document.querySelector<HTMLElement>('[data-workflow-context-menu][data-state="open"]')!;
    const dialog = document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]')!;
    await act(() => settleAnimations(menu));
    expectContained(menu, dialog);
    if (view === 'project') {
      await expectWrappedRenameHint(menu, dialog);
    }
    await act(() => userEvent.keyboard('{End}'));
    const lastAction = [...menu.querySelectorAll<HTMLElement>('[role="menuitem"]')].at(-1)!;
    await vi.waitFor(() => {
      expect(menu.getAttribute('aria-activedescendant')).toBe(lastAction.id);
      expectContained(lastAction, menu);
      expectContained(lastAction, dialog);
    });
    await resizeViewport(700, 320);
    // Let the grid take its new width while the menu is open, as a resize does outside the test's act scopes.
    await wait(50);
    await vi.waitFor(() => {
      expect(menu.getAttribute('data-state')).toBe('open');
      expectContained(menu, dialog);
    });
    await act(() => userEvent.keyboard('{Escape}'));
    // A wider grid can reflow the card into another row, which mounts it afresh: focus returns to the card by its id.
    await vi.waitFor(() => expect(document.activeElement?.id).toBe(source.id));
    expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).toBeNull();
  });

  it('auto-selects the first entry and moves the selection when it disappears', async () => {
    await openWith(LOADED_SNAPSHOT);

    expect(card('wf-portrait')?.getAttribute('aria-pressed')).toBe('true');
    expect(card('wf-landscape')?.getAttribute('aria-pressed')).toBe('false');

    await act(() => card('wf-landscape')?.click());

    expect(card('wf-landscape')?.getAttribute('aria-pressed')).toBe('true');

    await act(() => browse.setSnapshot?.(withSnapshot({ entries: [SKETCH, UPSCALE] })));

    expect(card('wf-sketch')?.getAttribute('aria-pressed')).toBe('true');
  });

  it('opens a workflow on double click', async () => {
    await openWith(LOADED_SNAPSHOT);

    await act(() => {
      card('wf-landscape')?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    });

    expect(loader.open).toHaveBeenCalledTimes(1);
    expect(loader.open).toHaveBeenCalledWith(LANDSCAPE.item, 'first-copy');
  });

  it('dedupes entries that share a workflow id', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, LANDSCAPE, { ...PORTRAIT }] }));

    expect(cards()).toHaveLength(2);
    expect(document.querySelectorAll('[data-workflow-card="wf-portrait"]')).toHaveLength(1);
  });

  it('keeps the loaded grid and reports a failed load-more inline', async () => {
    await openWith(withSnapshot({ error: 'Failed to load more workflows.', status: 'error' }));

    expect(cards()).toHaveLength(4);
    expect(document.body.textContent ?? '').toContain('Failed to load more workflows.');
    expect(document.body.textContent ?? '').not.toContain('No workflows match these filters.');
  });

  it('shows the loading copy on the very first paint, before the store leaves idle', async () => {
    // Initial idle state precedes loading and must not render as no matches.
    await openWith({
      entries: [],
      error: null,
      filter: { direction: 'desc', place: 'user', search: '', sort: 'updated', tags: [] },
      page: 0,
      pages: 0,
      status: 'idle',
      tagCounts: [],
      total: 0,
      placeTotals: {},
    });

    const text = document.body.textContent ?? '';

    expect(text).toContain('Loading workflows…');
    expect(text).not.toContain('No workflows match these filters.');
  });

  it('shows the empty state only when nothing loaded, saying why when nothing filters the place', async () => {
    await openWith(withSnapshot({ entries: [], status: 'loaded', total: 0 }));

    expect(cards()).toHaveLength(0);
    expect(document.body.textContent ?? '').toContain('No saved workflows yet.');

    await clickTag('portrait');
    expect(document.body.textContent ?? '').toContain('No workflows match these filters.');

    await clickPlace('recent');
    expect(document.body.textContent ?? '').toContain('Nothing opened yet.');
  });

  it('shows the selected workflow in the rail and follows the selection', async () => {
    await openWith(LOADED_SNAPSHOT);

    const detail = () => document.querySelector<HTMLElement>('[data-workflow-detail]');

    expect(detail()?.dataset.workflowDetail).toBe('wf-portrait');

    await act(() => card('wf-landscape')?.click());

    expect(detail()?.dataset.workflowDetail).toBe('wf-landscape');
    expect(detail()?.textContent).toContain('Landscape Pass');
  });

  it("opens a tile's actions from its own menu button, toggles it closed, and hands focus back to the button", async () => {
    await openWith(LOADED_SNAPSHOT);

    const button = document.querySelector<HTMLButtonElement>('[data-workflow-card-menu="wf-landscape"]');
    expect(button).not.toBeNull();
    // A sibling of the card button, never nested inside it.
    expect(document.querySelector('[data-workflow-card="wf-landscape"]')?.tagName).toBe('BUTTON');
    expect(document.querySelector('[data-workflow-card="wf-landscape"]')?.contains(button)).toBe(false);

    await act(async () => {
      button?.click();
      await settleFrame();
    });

    // The button opens the same actions the right-click offers, anchored to the button, and leaves the selection.
    expect(document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail).toBe('wf-portrait');
    const open = () =>
      document.querySelector<HTMLElement>('[data-workflow-context-menu][data-state="open"] [data-menu-item="open"]');
    await vi.waitFor(() => expect(open()).not.toBeNull());
    await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull(), { timeout: 2000 });

    await act(async () => {
      await userEvent.keyboard('{Escape}');
    });
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).toBeNull()
    );
    expect(document.activeElement).toBe(button);

    // A second press on the same button closes the menu instead of reopening it elsewhere. The press is a pointer
    // sequence, since the pointerdown lands outside the menu and must be kept from dismissing it first.
    await pressTileMenuButton(button);
    await vi.waitFor(() => expect(open()).not.toBeNull());
    expect(button?.getAttribute('aria-expanded')).toBe('true');
    await pressTileMenuButton(button);
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).toBeNull()
    );
    expect(button?.getAttribute('aria-expanded')).toBe('false');
  });

  it("moves the open menu to another tile's button and hands the expanded state over", async () => {
    await openWith(LOADED_SNAPSHOT);

    const first = document.querySelector<HTMLButtonElement>('[data-workflow-card-menu="wf-landscape"]');
    const second = document.querySelector<HTMLButtonElement>('[data-workflow-card-menu="wf-sketch"]');
    expect(second).not.toBeNull();

    await pressTileMenuButton(first);
    await vi.waitFor(() => expect(first?.getAttribute('aria-expanded')).toBe('true'));

    await pressTileMenuButton(second);

    await vi.waitFor(() => expect(second?.getAttribute('aria-expanded')).toBe('true'));
    expect(first?.getAttribute('aria-expanded')).toBe('false');
    expect(document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail).toBe('wf-portrait');
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')?.id).toBe(
        second?.getAttribute('aria-controls')
      )
    );
  });

  it('hands focus to the template rail menu it opens', async () => {
    await openWith(LOADED_SNAPSHOT);
    await act(() => card('wf-landscape')?.click());

    const trigger = document.querySelector<HTMLElement>('[data-workflow-detail] [aria-label="More actions"]');
    expect(trigger).not.toBeNull();
    await act(async () => {
      trigger?.click();
      await settleFrame();
    });

    await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull(), { timeout: 2000 });
  });

  describe('a card menu acts on its own card, never moving the selection', () => {
    const detail = () => document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail;
    const openMenu = () => document.querySelector<HTMLElement>('[data-workflow-context-menu][data-state="open"]');
    const alertDialog = () => document.querySelector<HTMLElement>('[role="alertdialog"]');
    const confirmIn = async (label: string) => {
      const confirm = [...(alertDialog()?.querySelectorAll('button') ?? [])].find(
        (candidate) => (candidate.textContent ?? '').trim() === label
      );
      expect(confirm).not.toBeUndefined();
      await act(async () => {
        confirm?.click();
        await settleFrame();
      });
    };

    /** Raises Landscape's menu by one of its three paths and returns the control that names it. */
    const raise = async (path: 'right-click' | 'Shift+F10' | 'More button') => {
      const source = card('wf-landscape')!;
      if (path === 'More button') {
        const more = document.querySelector<HTMLElement>('[data-workflow-card-menu="wf-landscape"]')!;
        await pressTileMenuButton(more);
        return more;
      }
      await act(() => source.focus());
      await act(() =>
        source.dispatchEvent(
          // A keyboard-raised menu (Shift+F10, the Menu key) reports no pointer position.
          path === 'Shift+F10'
            ? new MouseEvent('contextmenu', { bubbles: true, cancelable: true })
            : new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
        )
      );
      return source;
    };

    it.each(['right-click', 'Shift+F10', 'More button'] as const)(
      'opens from a %s on another card, keeping the rail, and renames that workflow',
      async (path) => {
        await openWith(LOADED_SNAPSHOT);
        await act(() => card('wf-upscale')?.click());
        expect(detail()).toBe('wf-upscale');

        const trigger = await raise(path);

        await vi.waitFor(() => expect(openMenu()?.id).toBe('workflow-library-card-wf-landscape-actions'));
        expect(trigger.getAttribute('aria-controls')).toBe(openMenu()?.id);
        expect(trigger.getAttribute('aria-expanded')).toBe('true');
        expect(detail()).toBe('wf-upscale');
        expect(card('wf-upscale')?.getAttribute('aria-pressed')).toBe('true');
        expect(card('wf-landscape')?.getAttribute('aria-pressed')).toBe('false');

        await act(async () => {
          openMenu()?.querySelector<HTMLElement>('[data-menu-item="rename"]')?.click();
          await settleFrame();
        });
        const input = await vi.waitFor(() => {
          const element = document.querySelector<HTMLInputElement>('input[name="renameValue"]');
          expect(element?.value).toBe('Landscape Pass');
          return element!;
        });
        const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
        await act(async () => {
          setValue?.call(input, 'Landscape, tuned');
          input.dispatchEvent(new Event('input', { bubbles: true }));
          await Promise.resolve();
        });
        await act(async () => {
          input.closest('form')?.querySelector<HTMLButtonElement>('button[type="submit"]')?.click();
          await settleFrame();
        });

        expect(libraryWrites.update).toHaveBeenCalledExactlyOnceWith(
          'wf-landscape',
          expect.objectContaining({ name: 'Landscape, tuned' }),
          expect.objectContaining({ expectedRevision: 3 })
        );
        expect(detail()).toBe('wf-upscale');
      }
    );

    it('runs Open on the card it was raised for', async () => {
      await openWith(LOADED_SNAPSHOT);
      await raise('right-click');

      const open = () => openMenu()?.querySelector<HTMLElement>('[data-menu-item="open"]');
      await vi.waitFor(() => expect(open()).not.toBeUndefined());
      await act(() => open()?.click());

      expect(loader.open).toHaveBeenCalledWith(LANDSCAPE.item, 'first-copy');
      expect(detail()).toBe('wf-portrait');
    });

    it('deletes another card and keeps the selection; deleting the selected one clears it', async () => {
      await openWith(LOADED_SNAPSHOT);
      await act(() => card('wf-upscale')?.click());
      await raise('right-click');

      await vi.waitFor(() => expect(openMenu()?.querySelector('[data-menu-item="delete"]')).not.toBeNull());
      await act(async () => {
        openMenu()?.querySelector<HTMLElement>('[data-menu-item="delete"]')?.click();
        await settleFrame();
      });
      await vi.waitFor(() => expect(alertDialog()?.textContent).toContain('Landscape Pass'));
      await confirmIn('Delete');

      expect(libraryWrites.delete).toHaveBeenCalledExactlyOnceWith('wf-landscape', expect.anything());
      // The listing refreshes without it, as the store does after the invalidation.
      await act(() => browse.setSnapshot?.(withSnapshot({ entries: [PORTRAIT, SKETCH, UPSCALE], total: 23 })));
      expect(card('wf-landscape')).toBeNull();
      expect(detail()).toBe('wf-upscale');

      // The selected card's own menu deletes it as the rail does: the selection clears to the first entry.
      await act(() =>
        card('wf-upscale')?.dispatchEvent(
          new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
        )
      );
      await vi.waitFor(() => expect(openMenu()?.querySelector('[data-menu-item="delete"]')).not.toBeNull());
      await act(async () => {
        openMenu()?.querySelector<HTMLElement>('[data-menu-item="delete"]')?.click();
        await settleFrame();
      });
      await vi.waitFor(() => expect(alertDialog()?.textContent).toContain('Upscale Chain'));
      await confirmIn('Delete');
      expect(libraryWrites.delete).toHaveBeenLastCalledWith('wf-upscale', expect.anything());
      await act(() => browse.setSnapshot?.(withSnapshot({ entries: [PORTRAIT, SKETCH], total: 22 })));
      expect(detail()).toBe('wf-portrait');
    });

    it('keeps a confirmation opened from a menu on its own workflow when a refresh drops that card', async () => {
      await openWith(LOADED_SNAPSHOT);
      await raise('right-click');
      await vi.waitFor(() => expect(openMenu()?.querySelector('[data-menu-item="delete"]')).not.toBeNull());
      await act(async () => {
        openMenu()?.querySelector<HTMLElement>('[data-menu-item="delete"]')?.click();
        await settleFrame();
      });
      await vi.waitFor(() => expect(alertDialog()?.textContent).toContain('Landscape Pass'));

      // A refresh drops the card (a re-sort, a newer record): the menu host loses its entry, not the dialog its target.
      await act(() => browse.setSnapshot?.(withSnapshot({ entries: [PORTRAIT, SKETCH, UPSCALE], total: 23 })));
      // Past any exit transition: the confirmation is still up, still naming its workflow.
      await wait(400);
      expect(alertDialog()?.getAttribute('data-state')).toBe('open');
      expect(alertDialog()?.textContent).toContain('Landscape Pass');
      await confirmIn('Cancel');
      await vi.waitFor(() => expect(alertDialog()).toBeNull());

      // The cancelled confirmation does not come back with the next writable card's menu.
      await act(() =>
        card('wf-upscale')?.dispatchEvent(
          new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
        )
      );
      await vi.waitFor(() => expect(openMenu()?.id).toBe('workflow-library-card-wf-upscale-actions'));
      await settleFrame();
      expect(alertDialog()).toBeNull();
      expect(libraryWrites.delete).not.toHaveBeenCalled();
    });

    it("offers Rename and Delete by the card's own access, not the selection's", async () => {
      identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
      const othersShared = { ...NON_OWNER_SHARED, item: { ...NON_OWNER_SHARED.item, workflow_id: 'wf-shared' } };
      await openWith(withSnapshot({ entries: [OWNER_PRIVATE, othersShared] }));
      const items = () =>
        [...(openMenu()?.querySelectorAll<HTMLElement>('[data-menu-item]') ?? [])].map((item) => item.dataset.menuItem);
      const raiseOn = async (workflowId: string) => {
        await act(() =>
          card(workflowId)?.dispatchEvent(
            new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
          )
        );
        await vi.waitFor(() => expect(openMenu()?.id).toBe(`workflow-library-card-${workflowId}-actions`));
      };

      // The account's own template is selected; another account's is not the account's to change.
      expect(detail()).toBe('wf-portrait');
      await raiseOn('wf-shared');
      expect(items()).toEqual(['open', 'duplicate', 'fork-into-project', 'download-json']);
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(openMenu()).toBeNull());

      // And the other way round: with the read-only one selected, the own card's menu still offers its writes.
      await act(() => card('wf-shared')?.click());
      expect(detail()).toBe('wf-shared');
      await raiseOn('wf-portrait');
      expect(items()).toContain('rename');
      expect(items()).toContain('delete');
    });

    it('duplicates the card it was raised for, once per press, and selects the copy', async () => {
      let finishCopy!: (workflowId: string) => void;
      libraryWrites.create.mockImplementationOnce(
        () =>
          new Promise<string>((resolve) => {
            finishCopy = resolve;
          })
      );
      await openWith(LOADED_SNAPSHOT);
      const duplicateItem = () => openMenu()?.querySelector<HTMLElement>('[data-menu-item="duplicate"]');
      const raiseOn = async (workflowId: string) => {
        await act(() =>
          card(workflowId)?.dispatchEvent(
            new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
          )
        );
        await vi.waitFor(() => expect(openMenu()?.id).toBe(`workflow-library-card-${workflowId}-actions`));
      };

      await raiseOn('wf-landscape');
      await act(async () => {
        duplicateItem()?.click();
        await settleFrame();
      });
      expect(libraryWrites.cached).toHaveBeenCalledExactlyOnceWith(
        'wf-landscape',
        expect.objectContaining({ expectedRevision: 1 })
      );
      await vi.waitFor(() => expect(libraryWrites.create).toHaveBeenCalledOnce());
      expect(libraryWrites.create.mock.calls[0]![0].name).toBe('Landscape Pass copy');

      // While that copy is in flight, the same workflow offers no second Duplicate; another workflow still does.
      await raiseOn('wf-landscape');
      expect(duplicateItem()?.hasAttribute('data-disabled')).toBe(true);
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(openMenu()).toBeNull());
      await raiseOn('wf-sketch');
      expect(duplicateItem()?.hasAttribute('data-disabled')).toBe(false);
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(openMenu()).toBeNull());

      await act(async () => {
        finishCopy('wf-landscape-copy');
        await settleFrame();
      });
      // Listed after the first entry, so only the selection, not the rail's first-entry fallback, shows the copy.
      const copy = entry('wf-landscape-copy', 'Landscape Pass copy', readyEnrichment(7, 'flux'));
      await act(() => browse.setSnapshot?.(withSnapshot({ entries: [PORTRAIT, LANDSCAPE, copy, SKETCH, UPSCALE] })));
      expect(detail()).toBe('wf-landscape-copy');
      expect(libraryWrites.create).toHaveBeenCalledOnce();
    });

    it('closes when its card leaves the listing', async () => {
      await openWith(LOADED_SNAPSHOT);
      await raise('right-click');
      await vi.waitFor(() => expect(openMenu()).not.toBeNull());

      await act(() => browse.setSnapshot?.(withSnapshot({ entries: [PORTRAIT, SKETCH, UPSCALE], total: 23 })));

      await vi.waitFor(() => expect(openMenu()).toBeNull());
      // Coming back later does not reopen it.
      await act(() => browse.setSnapshot?.(LOADED_SNAPSHOT));
      await settleFrame();
      expect(openMenu()).toBeNull();
      expect(card('wf-landscape')?.closest('[data-menu-open]')).toBeNull();
    });

    it('keeps one menu open: the rail’s More and a card’s menu close each other', async () => {
      await openWith(LOADED_SNAPSHOT);
      const railMore = document.querySelector<HTMLElement>('[data-workflow-detail] [aria-label="More actions"]')!;
      const isRailMenuOpen = () => railMore.getAttribute('aria-expanded') === 'true';

      await act(() => userEvent.click(railMore));
      await vi.waitFor(() => expect(isRailMenuOpen()).toBe(true));

      await act(() => userEvent.click(card('wf-landscape')!, { button: 'right' }));
      await vi.waitFor(() => expect(openMenu()?.id).toBe('workflow-library-card-wf-landscape-actions'));
      expect(isRailMenuOpen()).toBe(false);

      // One press, however quick: the card's menu closes before the rail's opens over it.
      await act(() => userEvent.click(railMore));
      await vi.waitFor(() => expect(isRailMenuOpen()).toBe(true));
      await vi.waitFor(() => expect(openMenu()).toBeNull());
      expect(card('wf-landscape')?.getAttribute('aria-expanded')).toBeNull();
      // The press that closed the card's menu keeps focus in the menu it opened, rather than back on the card.
      await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull());
      await settleFrame();
      expect(isRailMenuOpen()).toBe(true);
    });

    it('leaves focus in the field a press moved it to when that press closes the menu', async () => {
      await openWith(LOADED_SNAPSHOT);
      await raise('right-click');
      await vi.waitFor(() => expect(openMenu()).not.toBeNull());
      const search = document.querySelector<HTMLInputElement>(
        'input[placeholder="Search names, tags, or descriptions"]'
      )!;

      await act(() => userEvent.click(search));

      await vi.waitFor(() => expect(openMenu()).toBeNull());
      await settleFrame();
      expect(document.activeElement).toBe(search);

      // The next menu returns focus to its card again.
      await raise('right-click');
      await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull());
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(document.activeElement).toBe(card('wf-landscape')));
    });
  });

  // The question mounts while the menu that asked it is still closing; the menu's teardown must not dismiss it.
  it('keeps the copy question a card menu raised for a template the project already holds', async () => {
    await openWith(LOADED_SNAPSHOT);

    await act(() =>
      card('wf-portrait')?.dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
      )
    );

    const open = () => document.querySelector<HTMLElement>('[data-workflow-context-menu] [data-menu-item="open"]');

    await vi.waitFor(() => expect(open()).not.toBeNull());
    await act(async () => {
      open()?.click();
      await settleFrame();
      await settleFrame();
    });

    await vi.waitFor(() => expect(document.querySelector('[data-workflow-context-menu]')).toBeNull());
    expect(document.querySelector('[data-library-copy-choice]')).not.toBeNull();
    expect(loader.open).not.toHaveBeenCalled();
  });

  it('closes the card context menu on Escape and hands focus back to the card', async () => {
    await openWith(LOADED_SNAPSHOT);

    // A keyboard-raised menu reports no pointer position.
    await act(() =>
      card('wf-landscape')?.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }))
    );

    const menu = () => document.querySelector<HTMLElement>('[data-workflow-context-menu]');

    // The positioner places the menu a frame after it mounts; anchored inside the card, not at the viewport corner.
    await vi.waitFor(() =>
      expect(menu()?.getBoundingClientRect().left ?? 0).toBeGreaterThan(
        card('wf-landscape')!.getBoundingClientRect().left
      )
    );

    await act(async () => {
      await userEvent.keyboard('{Escape}');
    });

    await vi.waitFor(() => expect(menu()).toBeNull());
    await vi.waitFor(() => expect(document.activeElement).toBe(card('wf-landscape')));
  });

  it('opens the selected workflow from the rail, the keyboard-reachable path', async () => {
    await openWith(LOADED_SNAPSHOT);

    const railOpen = () => document.querySelector<HTMLButtonElement>('[data-workflow-detail-open]');

    // Portrait needs nothing installed, but the project already holds a copy of it: Open asks what to do.
    expect(railOpen()?.textContent).toBe('Open…');
    await act(() => railOpen()?.click());

    expect(workflowUiStore.getSnapshot().libraryCopyChoice?.item.workflow_id).toBe(PORTRAIT.item.workflow_id);
    expect(loader.open).not.toHaveBeenCalled();

    // Upscale has no copy yet (and nothing to install first), so Open adds one.
    await act(() => card('wf-upscale')?.click());
    expect(railOpen()?.textContent).toBe('Open');
    await act(() => railOpen()?.click());

    expect(loader.open).toHaveBeenCalledWith(UPSCALE.item, 'first-copy');
  });

  it('lists the selected template’s facts: callable, opened, copies here, updated and created', async () => {
    const hoursAgo = (hours: number) => new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();
    const portrait = {
      ...PORTRAIT,
      item: {
        ...PORTRAIT.item,
        call_saved_workflow_compatibility: { is_callable: true, message: null, reason: 'ok' },
        created_at: hoursAgo(24 * 3),
        opened_at: hoursAgo(2),
        revision: 7,
        updated_at: hoursAgo(5),
      },
    };
    await openWith(withSnapshot({ entries: [portrait, LANDSCAPE] }));

    const detail = () => document.querySelector<HTMLElement>('[data-workflow-detail]')!;
    expect(detail().textContent).toContain('Opened 2 hours ago');
    expect(detail().querySelector('[data-workflow-callable]')?.textContent).toBe('Callable');
    const facts = () => detail().querySelector<HTMLElement>('[data-workflow-detail-facts]')!.textContent;
    // The project's one copy of Portrait was loaded at revision 4.
    expect(facts()).toContain('In this project1 copy · revision 4');
    expect(facts()).toContain('Updated5 hours ago · revision 7');
    expect(facts()).toContain('Created3 days ago');

    await act(() => card('wf-landscape')?.click());
    expect(facts()).toContain('In this projectNone');
    expect(detail().querySelector('[data-workflow-callable]')).toBeNull();
  });

  it('badges the cards with the models their workflows still need', async () => {
    await openWith(LOADED_SNAPSHOT);

    expect(card('wf-landscape')?.textContent).toContain('1 model to install');
    expect(card('wf-portrait')?.textContent).not.toContain('to install');
    // The rail offers the install where it lists what the workflow requires.
    await act(() => card('wf-landscape')?.click());
    expect(buttonWithText('Install 1 model')?.closest('[data-workflow-detail]')).not.toBeNull();
  });

  it('records the workflow the rail asked to preview', async () => {
    await openWith(LOADED_SNAPSHOT);

    await clickText('Preview graph');

    // Task 8 mounts the preview dialog from this pending selection.
    expect(document.querySelector('[data-pending-preview="wf-portrait"]')).not.toBeNull();
  });

  it("mounts the lazy preview dialog with the entry's compiled graph and hides Invoke", async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    const preview = document.querySelector('[data-preview-dialog]');
    expect(preview).not.toBeNull();
    expect(preview?.getAttribute('data-graph-id')).toBe('wf-preview-fixture');
    expect(preview?.getAttribute('data-hide-invoke')).toBe('true');
    expect(preview?.getAttribute('data-source-label')).toBe('Preview Fixture');
    // The document's one `integer` node made it through compilation.
    expect(preview?.textContent).toContain('integer');
  });

  const quickAction = (workflowId: string, action: 'open' | 'preview') =>
    card(workflowId)
      ?.closest('[data-workflow-card-root]')
      ?.querySelector<HTMLButtonElement>(`[data-card-action="${action}"]`) ?? null;

  it('selects a small card from a real click at its centre after the pointer rests on it, never opening it', async () => {
    await openWith(LOADED_SNAPSHOT);
    const target = card('wf-upscale')!;
    const box = target.getBoundingClientRect();
    const centre = { x: box.width / 2, y: box.height / 2 };

    // Resting reveals the quick actions; none of them may sit under the card's centre.
    await act(() => userEvent.hover(target, { position: centre }));
    await vi.waitFor(() => expect(getComputedStyle(quickAction('wf-upscale', 'open')!).opacity).toBe('1'));
    const hit = document.elementFromPoint(box.left + centre.x, box.top + centre.y);
    expect(hit?.closest('[data-workflow-card]')).toBe(target);

    await act(() => userEvent.click(target, { position: centre }));
    expect(document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail).toBe('wf-upscale');
    expect(target.getAttribute('aria-pressed')).toBe('true');
    expect(loader.open).not.toHaveBeenCalled();
  });

  it('keeps hidden quick actions from taking the pointer until the card shows them', async () => {
    await openWith(LOADED_SNAPSHOT);
    await act(() => userEvent.unhover(card('wf-landscape')!));
    const open = () => quickAction('wf-landscape', 'open');
    const menu = document.querySelector<HTMLElement>('[data-workflow-card-menu="wf-landscape"]')!;

    // At rest the strip is transparent and lets presses through to the card beneath.
    expect(getComputedStyle(menu).pointerEvents).toBe('none');

    await act(() => userEvent.hover(card('wf-landscape')!));
    await vi.waitFor(() => expect(getComputedStyle(open()!).pointerEvents).toBe('auto'));
    await act(() => userEvent.click(open()!));
    expect(loader.open).toHaveBeenCalledWith(LANDSCAPE.item, 'first-copy');
  });

  it.each(['classic', 'light'])(
    'steps the opaque Preview and More quick actions on hover, and More while its menu is open (%s)',
    async (theme) => {
      // The default theme is the root's; others are named by its data-theme.
      if (theme !== 'classic') {
        document.documentElement.dataset.theme = theme;
        onTestFinished(() => {
          delete document.documentElement.dataset.theme;
        });
      }
      await openWith(LOADED_SNAPSHOT);
      const preview = () => quickAction('wf-landscape', 'preview')!;
      const more = document.querySelector<HTMLElement>('[data-workflow-card-menu="wf-landscape"]')!;

      // Over the card, the actions show on their panel fill.
      await act(() => userEvent.hover(card('wf-landscape')!));
      await vi.waitFor(() => expect(getComputedStyle(preview()).opacity).toBe('1'));
      await act(() => settleAnimations(preview()));
      const rest = getComputedStyle(preview()).backgroundColor;
      expect(getComputedStyle(more).backgroundColor).toBe(rest);

      await act(() => userEvent.hover(preview()));
      await vi.waitFor(() => expect(getComputedStyle(preview()).backgroundColor).not.toBe(rest));

      await act(() => userEvent.hover(more));
      await vi.waitFor(() => expect(getComputedStyle(more).backgroundColor).not.toBe(rest));

      await pressTileMenuButton(more);
      await vi.waitFor(() => expect(more.getAttribute('aria-expanded')).toBe('true'));
      // With the pointer on the open menu, not the button, the button still shows the state.
      const menu = await vi.waitFor(() => {
        const element = document.querySelector<HTMLElement>('[data-workflow-context-menu][data-state="open"]');
        expect(element).not.toBeNull();
        return element!;
      });
      await act(() => settleAnimations(menu));
      await act(() => userEvent.hover(menu));
      expect(more.matches(':hover')).toBe(false);
      await vi.waitFor(() => expect(getComputedStyle(more).backgroundColor).not.toBe(rest));
    }
  );

  it('opens from the Open quick action without taking focus, asking first when the project holds a copy', async () => {
    await openWith(LOADED_SNAPSHOT);

    // The project holds a copy of Portrait: the quick action says so and asks, as the rail does.
    await act(() => userEvent.hover(card('wf-portrait')!));
    expect(quickAction('wf-portrait', 'open')?.textContent).toBe('Open…');
    await act(() => userEvent.click(quickAction('wf-portrait', 'open')!));
    expect(workflowUiStore.getSnapshot().libraryCopyChoice?.item.workflow_id).toBe('wf-portrait');
    expect(loader.open).not.toHaveBeenCalled();
    // A pointer-only control hands focus to its card, so the question it raised returns focus there.
    await vi.waitFor(() => expect(document.querySelector('[data-library-copy-choice]')).not.toBeNull());
    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(document.activeElement).toBe(card('wf-portrait')));

    await act(() => userEvent.hover(card('wf-upscale')!));
    expect(quickAction('wf-upscale', 'open')?.textContent).toBe('Open');
    await act(() => userEvent.click(quickAction('wf-upscale', 'open')!));
    expect(loader.open).toHaveBeenCalledWith(UPSCALE.item, 'first-copy');
    expect(document.activeElement).toBe(card('wf-upscale'));
  });

  it('keeps Open and Preview out of the tab order: Tab goes from a card to its menu button', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));
    await act(() => userEvent.hover(card('wf-preview-fixture')!));
    expect(quickAction('wf-preview-fixture', 'open')?.tabIndex).toBe(-1);
    expect(quickAction('wf-preview-fixture', 'preview')?.tabIndex).toBe(-1);

    await act(() => card('wf-preview-fixture')!.focus());
    await act(() => userEvent.keyboard('{Tab}'));

    expect(document.activeElement?.getAttribute('data-workflow-card-menu')).toBe('wf-preview-fixture');
  });

  it('suppresses the native menu on the actions strip and menu button, and keeps an open menu for retargeting', async () => {
    await openWith(LOADED_SNAPSHOT);
    const menuButton = document.querySelector<HTMLElement>('[data-workflow-card-menu="wf-landscape"]')!;
    const strip = menuButton.closest<HTMLElement>('[data-workflow-card-actions]')!;

    const contextMenu = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 300, clientY: 300 });
    await act(() => menuButton.dispatchEvent(contextMenu));
    expect(contextMenu.defaultPrevented).toBe(true);
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).not.toBeNull()
    );
    expect(document.querySelector('[data-workflow-detail]')?.getAttribute('data-workflow-detail')).toBe('wf-portrait');

    // An open menu treats a right-click on a card's strip, or a press on a menu button, as a retarget, not a dismissal.
    const retargets = (target: Element, button: number) => {
      const preventDefault = vi.fn();
      const original = new PointerEvent('pointerdown', { bubbles: true, button });
      target.addEventListener(
        'pointerdown',
        (event) => keepCardMenuOpenForRetarget({ detail: { originalEvent: event }, preventDefault }),
        { once: true }
      );
      target.dispatchEvent(original);
      return preventDefault.mock.calls.length > 0;
    };
    expect(retargets(strip, 2)).toBe(true);
    expect(retargets(menuButton, 0)).toBe(true);
    expect(retargets(strip, 0)).toBe(false);
    expect(retargets(document.querySelector('[data-workflow-detail]')!, 2)).toBe(false);
  });

  it('keeps focus on the same workflow when a card-size change moves it to another row', async () => {
    await resizeViewport(1280, 800);
    const entries = Array.from({ length: 24 }, (_, index) => ({
      ...PORTRAIT,
      item: { ...PORTRAIT.item, workflow_id: `size-${index}` },
    }));
    await openWith({ ...LOADED_SNAPSHOT, entries, total: 24, pages: 2 });
    const before = card('size-11')!;
    await act(() => before.focus());
    const rowBefore = before.closest('[data-index]')?.getAttribute('data-index');

    await act(async () => {
      preferences.setSnapshot({ ...preferences.port.getSnapshot(), workflowLibraryCardSize: 'large' });
      await settleFrame();
    });

    await vi.waitFor(() => {
      const after = card('size-11')!;
      expect(after.closest('[data-index]')?.getAttribute('data-index')).not.toBe(rowBefore);
      expect(document.activeElement).toBe(after);
    });
  });

  it('pins the scroll area’s overflow properties above the cards, so scrolling restyles none of them', async () => {
    const entries = Array.from({ length: 60 }, (_, index) => ({
      ...PORTRAIT,
      item: { ...PORTRAIT.item, workflow_id: `pin-${index}` },
    }));
    await openWith({ ...LOADED_SNAPSHOT, entries, total: 60, pages: 3 });
    const viewport = gridViewport()!;
    await act(async () => {
      viewport.scrollTop = 400;
      viewport.dispatchEvent(new Event('scroll'));
      await settleFrame();
    });

    // Whatever custom properties the scroll area writes on its viewport, none may reach the cards changing.
    const viewportProperties = () => [...viewport.style].filter((name) => name.startsWith('--'));
    await vi.waitFor(() => expect(viewportProperties().length).toBeGreaterThan(0));
    const style = getComputedStyle(cards()[0]!);
    for (const name of viewportProperties()) {
      expect(['', '0px'], name).toContain(style.getPropertyValue(name).trim());
    }
  });

  it("previews an entry from its card's Preview quick action, selecting it", async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));
    const quickPreview = (workflowId: string) =>
      card(workflowId)
        ?.closest('[data-workflow-card-root]')
        ?.querySelector<HTMLButtonElement>('[data-card-action="preview"]');

    // Pointer-only quick actions mount on the card's first pointer visit, so mounting a row stays cheap.
    expect(quickPreview('wf-preview-fixture')).toBeNull();
    await act(() => userEvent.hover(card('wf-preview-fixture')!));
    expect(quickPreview('wf-preview-fixture')).not.toBeNull();
    await act(() => quickPreview('wf-preview-fixture')!.click());
    await waitForPreviewDialog();

    expect(document.querySelector('[data-workflow-detail]')?.getAttribute('data-workflow-detail')).toBe(
      'wf-preview-fixture'
    );
    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-graph-id')).toBe('wf-preview-fixture');
  });

  it('commits the first preview in the click that opens it once the library has loaded the dialog', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));
    await act(() => card('wf-preview-fixture')?.click());
    await vi.dynamicImportSettled();

    // A Suspense reveal here lets StrictMode replay the open dialog's effects after its focus trap started.
    act(() => (buttonWithText('Preview graph') as HTMLButtonElement).click());

    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-preview-open')).toBe('true');
  });

  it('disables the Preview action for an entry whose enrichment is not ready', async () => {
    await openWith(LOADED_SNAPSHOT);

    await act(() => card('wf-sketch')?.click());

    const button = buttonWithText('Preview graph') as HTMLButtonElement | undefined;
    expect(button?.disabled).toBe(true);
  });

  it('keeps the preview mounted through its exit transition, then resets the pending preview', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-preview-open')).toBe('true');

    await clickText('Close preview');

    // Keep the closing preview mounted until its exit animation finishes.
    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-preview-open')).toBe('false');
    expect(document.querySelector('[data-pending-preview]')).not.toBeNull();

    await clickText('Finish preview exit');

    expect(document.querySelector('[data-preview-dialog]')).toBeNull();
    expect(document.querySelector('[data-pending-preview]')).toBeNull();
  });

  it('keeps a re-opened preview mounted when the previous exit reports in late', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    await clickText('Close preview');
    // Previewing again mid-transition re-opens the same dialog…
    await clickText('Preview graph');

    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-preview-open')).toBe('true');

    // …and the exit report from the *previous* close must not unmount it.
    await clickText('Finish preview exit');

    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-preview-open')).toBe('true');
  });

  it('does not resurrect the preview after the library dialog closes and reopens', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    expect(document.querySelector('[data-preview-dialog]')).not.toBeNull();

    // Use the dialog's real Close path, then reopen through controlled props to verify reset and mount cleanup.
    const closeButton = document.querySelector<HTMLButtonElement>('button[aria-label="Close"]');
    expect(closeButton).not.toBeNull();
    // Settle deferred Dialog close transitions inside act.
    await act(async () => {
      closeButton?.click();
      await settleFrame();
    });

    await renderDialog(false);
    expect(document.querySelector('[data-preview-dialog]')).toBeNull();

    await renderDialog(true);
    expect(document.querySelector('[data-preview-dialog]')).toBeNull();
  });

  it('puts the title and Close in the header, the places in the rail and the toolbar over the grid', async () => {
    await resizeViewport(1280, 800);
    await openWith(LOADED_SNAPSHOT);

    const content = document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]')!;
    const title = content.querySelector<HTMLElement>('[data-part="title"]')!;
    const close = content.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!;
    const rail = content.querySelector<HTMLElement>('[data-workflow-library-rail]')!;
    const toolbar = content.querySelector<HTMLElement>('[data-workflow-library-toolbar]')!;
    const search = toolbar.querySelector<HTMLElement>('input[type="search"]')!;
    const closeBox = close.getBoundingClientRect();
    const titleBox = title.getBoundingClientRect();

    // Close shares the title's row at the dialog's end, above the rail and the toolbar.
    expect(closeBox.top + closeBox.height / 2).toBeGreaterThanOrEqual(titleBox.top);
    expect(closeBox.top + closeBox.height / 2).toBeLessThanOrEqual(titleBox.bottom);
    expect(closeBox.right).toBeGreaterThan(toolbar.getBoundingClientRect().right - 40);
    expect(closeBox.bottom).toBeLessThanOrEqual(toolbar.getBoundingClientRect().top);
    expect(rail.getBoundingClientRect().right).toBeLessThanOrEqual(toolbar.getBoundingClientRect().left);
    expect(toolbar.getBoundingClientRect().bottom).toBeLessThanOrEqual(gridViewport()!.getBoundingClientRect().top);
    // The search takes the toolbar's free width.
    expect(search.getBoundingClientRect().width).toBeGreaterThan(toolbar.getBoundingClientRect().width / 2);
    // The search is where typing starts.
    await vi.waitFor(() => expect(document.activeElement).toBe(search));
  });

  const NARROW_WINDOWS = [
    [900, 700],
    [640, 400],
  ] as const;

  it.each(NARROW_WINDOWS)(
    'collapses the places, tags and order controls into the toolbar in a %i×%i window',
    async (width, height) => {
      await resizeViewport(width, height);
      await openWith(withSnapshot({ placeTotals: { default: 25, recent: 3, user: 4 } }));
      browse.setWorkflowLibraryBrowseFilter.mockClear();

      const content = document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]')!;
      const toolbar = content.querySelector<HTMLElement>('[data-workflow-library-toolbar]')!;
      const search = toolbar.querySelector<HTMLElement>('input[type="search"]')!;
      const placeSelect = toolbar.querySelector<HTMLElement>('[data-workflow-library-place-select] [role="combobox"]')!;
      const options = toolbar.querySelector<HTMLButtonElement>('[data-workflow-library-narrow-options]')!;
      const tagsButton = () => content.querySelector<HTMLButtonElement>('[data-workflow-library-tags-trigger]')!;
      const results = () => content.querySelector('[data-workflow-library-results]')?.textContent;

      // The rail and the wide layout's order, size and view controls leave the layout; nothing scrolls sideways.
      for (const wide of [
        content.querySelector('[data-workflow-library-rail]'),
        sortTrigger(),
        content.querySelector('button[aria-label="View options"]'),
      ]) {
        expect(wide).not.toBeNull();
        expect(isShown(wide)).toBe(false);
      }
      expect(content.scrollWidth).toBeLessThanOrEqual(content.clientWidth + 1);
      expect(toolbar.scrollWidth).toBeLessThanOrEqual(toolbar.clientWidth + 1);
      for (const control of [placeSelect, search, options, tagsButton()]) {
        expect(isShown(control)).toBe(true);
        expectContained(control, content);
      }
      // The places select leads the toolbar and names the place with its count.
      expect(placeSelect.getBoundingClientRect().right).toBeLessThanOrEqual(search.getBoundingClientRect().left);
      expect(placeSelect.textContent).toBe('My library4');

      // Sort, direction, card size and fit share one menu, which stays open across choices.
      await act(() => userEvent.click(options));
      await vi.waitFor(() => expect(options.getAttribute('aria-expanded')).toBe('true'));
      const menu = document.getElementById(options.getAttribute('aria-controls') ?? '')!;
      const radio = (label: string) =>
        [...menu.querySelectorAll<HTMLElement>('[role="menuitemradio"]')].find((item) => item.textContent === label)!;
      expect([...menu.querySelectorAll('[role="menuitemradio"]')].map((item) => item.textContent)).toEqual([
        'Last opened',
        'Last updated',
        'Last run',
        'Date created',
        'Name',
        'Newest first',
        'Oldest first',
        'Grid',
        'List',
        'Small cards',
        'Large cards',
        'Fill',
        'Fit',
      ]);
      expect(radio('Last updated').getAttribute('aria-checked')).toBe('true');
      expect(radio('Newest first').getAttribute('aria-checked')).toBe('true');
      await act(() => userEvent.click(radio('Name')));
      expect(setLibraryPreferences).toHaveBeenCalledWith({
        workflowLibrarySort: 'name',
        workflowLibrarySortDirection: 'asc',
      });
      await act(() => userEvent.click(radio('Oldest first')));
      expect(setLibraryPreferences).toHaveBeenCalledWith({ workflowLibrarySortDirection: 'asc' });
      await act(() => userEvent.click(radio('Large cards')));
      expect(setLibraryPreferences).toHaveBeenCalledWith({ workflowLibraryCardSize: 'large' });
      expect(menu.isConnected && isShown(menu)).toBe(true);
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(document.activeElement).toBe(options));

      // Tags open in a popover; with the rail gone, the Tags button counts the checked tags and the result line names
      // them. Nothing else is added under the toolbar.
      expect(tagsButton().textContent).toBe('Tags');
      await act(() => userEvent.click(tagsButton()));
      await vi.waitFor(() => expect(tagsButton().getAttribute('aria-expanded')).toBe('true'));
      const popover = document.getElementById(tagsButton().getAttribute('aria-controls') ?? '')!;
      expect(popover.textContent).toContain('Showing workflows with any checked tag');
      await act(() => userEvent.click(popover.querySelector<HTMLElement>('[data-library-tag="portrait"]')!));
      await act(() => userEvent.click(popover.querySelector<HTMLElement>('[data-library-tag="upscale"]')!));
      expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenLastCalledWith({ tags: ['portrait', 'upscale'] });
      const resultLine = content.querySelector<HTMLElement>('[data-workflow-library-results]')!;
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(document.activeElement).toBe(tagsButton()));
      expect(tagsButton().textContent).toBe('Tags2');
      expect(results()).toBe('24 workflows · portrait or upscale');
      expect(toolbar.nextElementSibling).toBe(resultLine);

      // Clear inside the popover keeps focus on its first tag, and closing it returns focus to the Tags button.
      await act(() => userEvent.click(tagsButton()));
      await vi.waitFor(() => expect(tagsButton().getAttribute('aria-expanded')).toBe('true'));
      const reopened = document.getElementById(tagsButton().getAttribute('aria-controls') ?? '')!;
      await act(() =>
        userEvent.click([...reopened.querySelectorAll('button')].find((b) => b.textContent === 'Clear')!)
      );
      expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenLastCalledWith({ tags: [] });
      expect(document.activeElement).toBe(
        reopened.querySelector('[data-library-tag="portrait"] input[type="checkbox"]')
      );
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(document.activeElement).toBe(tagsButton()));
      expect(results()).toBe('24 workflows');

      // Choosing a place from the select moves there, keeping the search and dropping the tags.
      await act(() => userEvent.click(placeSelect));
      const templates = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find(
        (option) => option.textContent === 'Templates25'
      );
      expect(templates).not.toBeUndefined();
      await act(() => userEvent.click(templates!));
      expect(workflowUiStore.getSnapshot().libraryTab).toBe('default');
      expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenLastCalledWith(
        expect.objectContaining({ place: 'default', tags: [] })
      );
      expect(placeSelect.textContent).toBe('Templates25');
    }
  );

  it.each(NARROW_WINDOWS)(
    'shows the selection as a bottom sheet whose details open over the grid in a %i×%i window',
    async (width, height) => {
      const entries = [
        PORTRAIT,
        ...Array.from({ length: 40 }, (_, index) => ({
          ...LANDSCAPE,
          item: { ...LANDSCAPE.item, workflow_id: `sheet-${index}` },
        })),
      ];
      await resizeViewport(width, height);
      await openWith(withSnapshot({ entries, total: entries.length, pages: 1 }));

      const content = document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]')!;
      const sheet = content.querySelector<HTMLElement>('[data-workflow-detail="wf-portrait"]')!;
      const summary = sheet.querySelector<HTMLElement>('[data-detail-sheet-summary]')!;
      const details = sheet.querySelector<HTMLElement>('[data-detail-sheet-details]')!;
      const toggle = summary.querySelector<HTMLButtonElement>('[data-detail-sheet-toggle]')!;
      const more = summary.querySelector<HTMLButtonElement>('button[aria-label="More actions"]')!;
      const grid = gridViewport()!;
      const search = content.querySelector<HTMLElement>('[data-workflow-library-toolbar] input[type="search"]')!;
      // The column that scrolls in a short window: the sheet's nearest scrolling ancestor.
      let centre = sheet.parentElement!;
      while (!/(auto|scroll)/.test(getComputedStyle(centre).overflowY)) {
        centre = centre.parentElement!;
      }

      // Collapsed: a summary row with the name, facts, Open and More under the grid; the full rail is out of the
      // layout and the tab order.
      expect(isShown(summary)).toBe(true);
      expect(isShown(details)).toBe(false);
      expectContained(summary, content);
      expect(summary.getBoundingClientRect().left).toBeGreaterThanOrEqual(grid.getBoundingClientRect().left - 1);
      expect(summary.getBoundingClientRect().bottom).toBeGreaterThan(grid.getBoundingClientRect().top);
      expect(summary.textContent).toContain('Portrait Studio');
      expect(summary.querySelector('[data-detail-sheet-open]')?.textContent).toBe('Open…');
      expect(toggle.textContent).toBe('Details and requirements');
      expect(toggle.getAttribute('aria-expanded')).toBe('false');
      expect(toggle.getAttribute('aria-controls')).toBe(details.id);

      // The summary's More offers the rail's actions.
      await act(() => userEvent.click(more));
      await vi.waitFor(() => expect(more.getAttribute('aria-expanded')).toBe('true'));
      const actions = document.getElementById(more.getAttribute('aria-controls') ?? '')!;
      expect(actions.querySelector('[data-menu-item="duplicate"]')).not.toBeNull();
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(document.activeElement).toBe(more));

      await act(() => {
        grid.scrollTop = 200;
        grid.dispatchEvent(new Event('scroll'));
      });
      const scrollTop = grid.scrollTop;
      expect(scrollTop).toBeGreaterThan(0);
      // From the top of the column, as the library opens; the pointer would scroll the toggle into view first.
      await act(() => {
        centre.scrollTop = 0;
        toggle.focus({ preventScroll: true });
      });
      const centreScrollTop = centre.scrollTop;

      // Expanded: the full rail covers the grid, which keeps its scroll position but leaves the tab order.
      await act(() => userEvent.keyboard('{Enter}'));
      expect(toggle.getAttribute('aria-expanded')).toBe('true');
      expect(toggle.textContent).toBe('Hide details');
      expect(isShown(details)).toBe(true);
      const detailsBox = details.getBoundingClientRect();
      const gridBox = grid.getBoundingClientRect();
      expect(detailsBox.top).toBeLessThanOrEqual(gridBox.top + 1);
      expect(detailsBox.left).toBeLessThanOrEqual(gridBox.left + 1);
      expect(detailsBox.right).toBeGreaterThanOrEqual(gridBox.right - 1);
      expect(getComputedStyle(grid).visibility).toBe('hidden');
      expect(grid.scrollTop).toBe(scrollTop);
      expect(details.contains(document.activeElement)).toBe(true);
      if (height <= 400) {
        // The short window's column scrolls the details into view.
        expect(centre.scrollTop).toBeGreaterThan(centreScrollTop);
      }
      // Compact: the summary heads the details with the full name; no large preview or second name above.
      expect(getComputedStyle(summary.querySelector('p')!).whiteSpace).not.toBe('nowrap');
      expect(
        [...details.querySelectorAll('p')].some((text) => text.textContent === 'Portrait Studio' && isShown(text))
      ).toBe(false);
      expect([...details.querySelectorAll('img')].some(isShown)).toBe(false);
      expect(details.querySelector('[data-requirement-status], [data-workflow-detail-facts]')).not.toBeNull();
      expect(
        [...details.querySelectorAll('button')].some(
          (button) => button.textContent === 'Preview graph' && isShown(button)
        )
      ).toBe(true);

      // A menu over the expanded sheet takes Escape for itself.
      await act(() => userEvent.click(more));
      await vi.waitFor(() => expect(more.getAttribute('aria-expanded')).toBe('true'));
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(more.getAttribute('aria-expanded')).toBe('false'));
      expect(toggle.getAttribute('aria-expanded')).toBe('true');

      // Escape from inside the sheet collapses it and returns focus to the disclosure, leaving the library open.
      await act(() => details.focus());
      await act(() => userEvent.keyboard('{Escape}'));
      expect(toggle.getAttribute('aria-expanded')).toBe('false');
      expect(document.activeElement).toBe(toggle);
      expect(isShown(details)).toBe(false);
      expect(getComputedStyle(grid).visibility).toBe('visible');
      expect(grid.scrollTop).toBe(scrollTop);
      // Expanding may scroll a short window's column to the details; collapsing scrolls it back to the toolbar.
      expect(centre.scrollTop).toBe(centreScrollTop);
      expectContained(search, content);
      expect(onOpenChange).not.toHaveBeenCalledWith(false);

      // Collapsed, Escape closes the library as usual.
      await act(() => userEvent.keyboard('{Escape}'));
      expect(onOpenChange).toHaveBeenCalledWith(false);
    }
  );

  it('wraps a long workflow name in full instead of scrolling the rail sideways', async () => {
    await openWith(withSnapshot({ entries: [LONG_NAMED, PORTRAIT] }));

    const detail = document.querySelector<HTMLElement>('[data-workflow-detail]');
    const body = detail?.parentElement;
    expect(detail?.dataset.workflowDetail).toBe('wf-long-name');
    expect(body).not.toBeNull();

    // 18rem rail plus its 1px borders. Anything wider is the rail growing to
    // its content's min-content width.
    expect(detail?.getBoundingClientRect().width).toBeLessThanOrEqual(18 * 16 + 2);
    expect(body?.scrollWidth).toBeLessThanOrEqual((body?.clientWidth ?? 0) + 1);

    // The rail's own (vertical) scroll area must not scroll horizontally: it
    // renders no horizontal scrollbar, so overflow there just hides content.
    const railViewport = detail?.querySelector<HTMLElement>(`[role="region"][aria-label="${LONG_NAME}"]`);
    expect(railViewport).not.toBeNull();
    expect(railViewport?.scrollWidth).toBeLessThanOrEqual((railViewport?.clientWidth ?? 0) + 1);

    // The detail rail must wrap the full workflow name without clipping.
    const heading = [...(detail?.querySelectorAll('p') ?? [])].find((element) => element.textContent === LONG_NAME);
    expect(heading, 'the full name should be rendered').not.toBeUndefined();
    expect(heading?.scrollWidth).toBeLessThanOrEqual((heading?.clientWidth ?? 0) + 1);
    // Wrapped, not squeezed onto one line.
    expect(heading?.getBoundingClientRect().height ?? 0).toBeGreaterThan(20);
  });

  describe('list layout', () => {
    const row = (workflowId: string) =>
      document.getElementById(`workflow-library-card-${workflowId}`)?.closest<HTMLElement>('[data-list-row]') ?? null;
    const primary = (workflowId: string) => row(workflowId)?.querySelector<HTMLElement>('[data-list-primary]') ?? null;
    const rail = () => document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail;
    const menu = () => document.querySelector<HTMLElement>('[data-workflow-context-menu][data-state="open"]');
    const openAsList = async (snapshot: WorkflowLibraryBrowseSnapshot) => {
      preferences.setSnapshot({ ...preferences.port.getSnapshot(), workflowLibraryLayout: 'list' });
      await openWith(snapshot);
      await vi.waitFor(() => expect(row('wf-portrait')).not.toBeNull());
    };

    it('starts as the grid, switches to rows from the view menu and keeps the choice with the account', async () => {
      await openWith(LOADED_SNAPSHOT);
      const item = (label: string) =>
        [...document.querySelectorAll<HTMLElement>('[role="menuitemradio"]')].find(
          (candidate) => candidate.textContent === label
        );

      expect(cards()).toHaveLength(4);
      expect(document.querySelector('[data-workflow-library-list]')).toBeNull();
      await act(() => userEvent.click(document.querySelector<HTMLElement>('button[aria-label="View options"]')!));
      await vi.waitFor(() => expect(item('List')).not.toBeUndefined());
      expect(item('Grid')?.getAttribute('aria-checked')).toBe('true');
      await act(() => userEvent.click(item('List')!));

      expect(setLibraryPreferences).toHaveBeenCalledWith({ workflowLibraryLayout: 'list' });
      await vi.waitFor(() => expect(row('wf-portrait')).not.toBeNull());
      expect(cards()).toHaveLength(0);
      expect(row('wf-landscape')?.textContent).toContain('Landscape Pass');
      expect(row('wf-landscape')?.textContent).toContain('1 model to install');
      expect(
        row('wf-landscape')?.querySelector('[data-workflow-readiness]')?.getAttribute('data-workflow-readiness')
      ).toBe('models');
      // The project holds a copy of Portrait; the server's 20 positions past the window wait as placeholders.
      expect(row('wf-portrait')?.querySelector('[data-workflow-in-project]')).not.toBeNull();
      expect(row('wf-sketch')?.querySelector('[data-enrichment-placeholder]')).not.toBeNull();
      expect(
        document
          .querySelector('[data-workflow-library-list] [role="list"]')
          ?.querySelector('[aria-setsize]')
          ?.getAttribute('aria-setsize')
      ).toBe('24');
      // Card size and thumbnail display shape only the grid.
      expect(item('Small cards')).toBeUndefined();
      expect(item('Fill')).toBeUndefined();
    });

    it('hides the list, keeping its scroll position, under the narrow sheet’s expanded details', async () => {
      await resizeViewport(900, 700);
      const entries = Array.from({ length: 30 }, (_, index) => ({
        ...LANDSCAPE,
        item: { ...LANDSCAPE.item, workflow_id: `sheet-${index}`, name: `Sheet ${index}` },
      }));
      await openAsList(withSnapshot({ entries: [PORTRAIT, ...entries], pages: 1, total: 31 }));
      const list = document.querySelector<HTMLElement>('[data-workflow-library-list]')!;
      const scroll = list.querySelector<HTMLElement>('[data-list-viewport]')!;
      await act(async () => {
        scroll.scrollTop = 300;
        scroll.dispatchEvent(new Event('scroll'));
        await settleFrame();
      });
      const toggle = document.querySelector<HTMLButtonElement>('[data-detail-sheet-toggle]')!;
      expect(getComputedStyle(list).visibility).toBe('visible');

      await act(() => toggle.focus({ preventScroll: true }));
      await act(() => userEvent.keyboard('{Enter}'));
      expect(toggle.getAttribute('aria-expanded')).toBe('true');
      expect(getComputedStyle(list).visibility).toBe('hidden');
      expect(scroll.scrollTop).toBe(300);

      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(getComputedStyle(list).visibility).toBe('visible'));
    });

    it('selects on a press, opens on a double-click, and moves the selection with ArrowDown and Enter', async () => {
      await openAsList(LOADED_SNAPSHOT);

      await act(() => userEvent.click(primary('wf-landscape')!));
      expect(rail()).toBe('wf-landscape');
      expect(primary('wf-landscape')?.getAttribute('aria-current')).toBe('true');

      await act(() => {
        primary('wf-landscape')?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
      });
      expect(loader.open).toHaveBeenCalledWith(LANDSCAPE.item, 'first-copy');

      await act(() => userEvent.keyboard('{ArrowDown}'));
      expect(document.activeElement).toBe(primary('wf-sketch'));
      await act(() => userEvent.keyboard('{Enter}'));
      expect(rail()).toBe('wf-sketch');
    });

    it('raises the same actions menu from a right-click and the More button, handing focus back to its source', async () => {
      await openAsList(LOADED_SNAPSHOT);

      await act(() => primary('wf-landscape')?.focus());
      await act(() =>
        primary('wf-landscape')?.dispatchEvent(
          new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 300, clientY: 200 })
        )
      );
      // The menu is the row's, not the selection's: the rail stays on the first entry.
      expect(rail()).toBe('wf-portrait');
      await vi.waitFor(() => expect(menu()?.querySelector('[data-menu-item="open"]')).not.toBeNull());
      // The row's button names the menu, so the dialog's focus trap lets focus into it.
      expect(primary('wf-landscape')?.getAttribute('aria-expanded')).toBe('true');
      expect(primary('wf-landscape')?.getAttribute('aria-controls')).toBe(menu()?.id);
      await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull(), {
        timeout: 2000,
      });
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(menu()).toBeNull());
      await vi.waitFor(() => expect(document.activeElement).toBe(primary('wf-landscape')));

      const more = row('wf-sketch')?.querySelector<HTMLButtonElement>('[data-workflow-card-menu]') ?? null;
      await pressTileMenuButton(more);
      expect(rail()).toBe('wf-portrait');
      await vi.waitFor(() => expect(menu()?.querySelector('[data-menu-item="open"]')).not.toBeNull());
      expect(more?.getAttribute('aria-expanded')).toBe('true');
      expect(more?.getAttribute('aria-controls')).toBe(menu()?.id);
      expect(primary('wf-sketch')?.hasAttribute('aria-expanded')).toBe(false);
      await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull(), {
        timeout: 2000,
      });
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(menu()).toBeNull());
      await vi.waitFor(() => expect(document.activeElement).toBe(more));
    });

    it("deletes another row from that row's menu and keeps the selection", async () => {
      await openAsList(LOADED_SNAPSHOT);
      await act(() => userEvent.click(primary('wf-upscale')!));
      expect(rail()).toBe('wf-upscale');

      await act(() => userEvent.click(primary('wf-landscape')!, { button: 'right' }));
      await vi.waitFor(() => expect(menu()?.id).toBe('workflow-library-card-wf-landscape-actions'));
      expect(rail()).toBe('wf-upscale');
      expect(primary('wf-upscale')?.getAttribute('aria-current')).toBe('true');

      await act(async () => {
        menu()?.querySelector<HTMLElement>('[data-menu-item="delete"]')?.click();
        await settleFrame();
      });
      const alert = await vi.waitFor(() => {
        const element = document.querySelector<HTMLElement>('[role="alertdialog"]');
        expect(element?.textContent).toContain('Landscape Pass');
        return element!;
      });
      await act(async () => {
        [...alert.querySelectorAll('button')].find((button) => button.textContent?.trim() === 'Delete')?.click();
        await settleFrame();
      });

      expect(libraryWrites.delete).toHaveBeenCalledExactlyOnceWith('wf-landscape', expect.anything());
      await act(() => browse.setSnapshot?.(withSnapshot({ entries: [PORTRAIT, SKETCH, UPSCALE], total: 23 })));
      await vi.waitFor(() => expect(row('wf-landscape')).toBeNull());
      expect(rail()).toBe('wf-upscale');
    });

    it('opens from the Open quick action without taking focus off the row, asking first when the project holds a copy', async () => {
      await openAsList(LOADED_SNAPSHOT);

      await act(() => userEvent.hover(primary('wf-portrait')!));
      const quickOpen = row('wf-portrait')?.querySelector<HTMLElement>('[data-row-action="open"]');
      expect(quickOpen?.textContent).toBe('Open…');
      expect(quickOpen?.tabIndex).toBe(-1);
      await act(() => userEvent.click(quickOpen!));

      expect(workflowUiStore.getSnapshot().libraryCopyChoice?.item.workflow_id).toBe('wf-portrait');
      expect(loader.open).not.toHaveBeenCalled();
      // The question returns focus to the row it was raised from.
      await vi.waitFor(() => expect(document.querySelector('[data-library-copy-choice]')).not.toBeNull());
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(document.activeElement).toBe(primary('wf-portrait')));

      await act(() => userEvent.hover(primary('wf-upscale')!));
      const upscaleOpen = row('wf-upscale')?.querySelector<HTMLElement>('[data-row-action="open"]');
      expect(upscaleOpen?.textContent).toBe('Open');
      await act(() => userEvent.click(upscaleOpen!));
      expect(loader.open).toHaveBeenCalledWith(UPSCALE.item, 'first-copy');
      expect(document.activeElement).toBe(primary('wf-upscale'));
    });
  });

  it('shows the busy overlay while a workflow is being applied', async () => {
    loader.phase.current = 'applying';

    await openWith(LOADED_SNAPSHOT);

    const status = document.querySelector('[role="status"]');
    expect(status?.textContent).toContain('Applying workflow…');
    expect(document.querySelector('[aria-busy="true"]')).not.toBeNull();
  });
});

describe('WorkflowLibraryDialog — This project', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onOpenChange: (isOpen: boolean) => void;

  const settleFrame = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

  /** `storeBound` opens and closes the dialog through the UI store, as the workflow dialog host does. */
  const renderDialog = async ({ storeBound = false }: { storeBound?: boolean } = {}) => {
    await act(async () => {
      root.render(
        <StrictMode>
          <QueryClientProvider client={queryClient}>
            <ChakraProvider value={system}>
              <WorkflowUiProvider adapter={UI_ADAPTER}>
                <GalleryHostProvider host={GALLERY_HOST}>
                  <WorkflowGraphPreviewProvider adapter={GRAPH_PREVIEW}>
                    {storeBound ? (
                      <StoreBoundLibraryDialog />
                    ) : (
                      <WorkflowLibraryDialog isOpen onOpenChange={onOpenChange} />
                    )}
                  </WorkflowGraphPreviewProvider>
                </GalleryHostProvider>
              </WorkflowUiProvider>
            </ChakraProvider>
          </QueryClientProvider>
        </StrictMode>
      );
      await settleFrame();
    });
  };

  const cards = () => [...document.querySelectorAll<HTMLElement>('[data-workflow-card]')];
  const card = (workflowId: string) => document.querySelector<HTMLElement>(`[data-workflow-card="${workflowId}"]`);
  const rail = () => document.querySelector<HTMLElement>('[data-project-workflow-detail]');
  const buttonWithText = (text: string) =>
    [...document.querySelectorAll('button')].find((candidate) => (candidate.textContent ?? '') === text);
  const alertDialog = () => document.querySelector<HTMLElement>('[role="alertdialog"]');

  const click = async (element: HTMLElement | null | undefined) => {
    expect(element).not.toBeFalsy();

    await act(async () => {
      element?.click();
      await settleFrame();
    });
  };
  const clickText = (text: string) => click(buttonWithText(text));
  // The grid can render its cards a frame after the dialog settles on a loaded CI worker.
  const selectCard = async (workflowId: string) => {
    await vi.waitFor(() => expect(card(workflowId)).not.toBeNull());
    await click(card(workflowId));
  };

  const menuItem = (value: string) => document.querySelector<HTMLElement>(`[data-menu-item="${value}"]`);
  const railMenuTrigger = () => rail()?.querySelector<HTMLElement>('[aria-label="More actions"]');
  // The trigger's data-state belongs to its tooltip; the menu's own state is aria-expanded. The menu opens on a
  // later frame, so read it only once its items are in the document.
  const openRailMenu = async () => {
    const trigger = railMenuTrigger();
    expect(trigger).not.toBeNull();

    if (trigger?.getAttribute('aria-expanded') !== 'true') {
      await click(trigger);
    }

    await vi.waitFor(
      () => {
        expect(railMenuTrigger()?.getAttribute('aria-expanded')).toBe('true');
        expect(document.querySelector('[data-menu-item]')).not.toBeNull();
      },
      { timeout: 2000 }
    );
  };
  // Escape only reaches the menu once it holds focus, which lands a frame later; toggling the trigger closes it
  // regardless. A closed menu keeps its items mounted, so aria-expanded is what says it has closed.
  const closeRailMenu = async () => {
    if (railMenuTrigger()?.getAttribute('aria-expanded') === 'true') {
      await click(railMenuTrigger());
    }

    await vi.waitFor(() => expect(railMenuTrigger()?.getAttribute('aria-expanded')).toBe('false'), { timeout: 2000 });
  };
  const clickRailMenuItem = async (value: string) => {
    await openRailMenu();
    await click(menuItem(value));
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    onOpenChange = vi.fn((_isOpen: boolean) => {});
    browse.ensureWorkflowLibraryBrowseLoaded.mockClear();
    browse.loadWorkflowLibraryPlaceTotals.mockClear();
    browse.setWorkflowLibraryBrowseFilter.mockClear();
    loader.open.mockClear();
    loader.replace.mockClear();
    loader.resume.mockClear();
    queryClient.clear();
    templateRecords.clear();
    loader.phase.current = 'idle';
    for (const command of Object.values(COMMANDS)) {
      command.mockClear();
    }
    project.setSnapshot(projectSnapshot(PROJECT_WORKFLOWS));
    preferences.setSnapshot(LIBRARY_PREFERENCES as unknown as WorkflowPreferences);
    browse.setSnapshot?.(LOADED_SNAPSHOT);
    accountLifecycle.invalidate();
    // The library opens on the project's own workflows, as it does from the workflow header.
    openWorkflowLibraryAtProjectWorkflow(PROJECT_ID, 'wf-alpha');
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('lists every project workflow as a card, badges the active one, and never fetches templates', async () => {
    await renderDialog();

    expect(document.querySelector('[data-library-tab="project"]')).not.toBeNull();
    expect(cards().map((element) => element.dataset.workflowCard)).toEqual(['wf-alpha', 'wf-bundled', 'wf-user']);
    expect(document.querySelector('[data-workflow-library-results]')?.textContent).toBe('3 workflows');

    const badges = [...document.querySelectorAll<HTMLElement>('[data-workflow-card] [data-active-workflow]')];
    expect(badges).toHaveLength(1);
    expect(badges[0]?.closest<HTMLElement>('[data-workflow-card]')?.dataset.workflowCard).toBe('wf-alpha');
    expect(badges[0]?.textContent).toBe('Active');

    // The project view is local; no library page is requested. The rail's totals are the only library reads.
    expect(browse.ensureWorkflowLibraryBrowseLoaded).not.toHaveBeenCalled();
    expect(browse.loadWorkflowLibraryPlaceTotals).toHaveBeenCalled();
    // The project's workflows carry no tags, so the rail lists none; its footer explains autosave.
    expect(document.querySelector('[data-library-tag]')).toBeNull();
    expect(document.querySelector('[data-workflow-library-rail]')?.textContent).toContain(
      'Edits autosave here; the library changes only when you save to it.'
    );
    expect(document.querySelector('[data-workflow-card] [data-workflow-visibility]')).toBeNull();
    expect(document.body.textContent).not.toContain('Public workflow');

    // The active workflow is already open: its card offers no Open quick action, as its rail disables Open.
    const quickOpen = (workflowId: string) =>
      card(workflowId)?.closest('[data-workflow-card-root]')?.querySelector('[data-card-action="open"]') ?? null;
    await act(() => userEvent.hover(card('wf-alpha')!));
    await act(() => userEvent.hover(card('wf-user')!));
    expect(quickOpen('wf-alpha')).toBeNull();
    expect(quickOpen('wf-user')).not.toBeNull();
  });

  it('lists the project’s workflows as rows: Active on the open one, which offers no Open, and a run date', async () => {
    preferences.setSnapshot({ ...preferences.port.getSnapshot(), workflowLibraryLayout: 'list' });
    await renderDialog();
    const row = (workflowId: string) =>
      document.querySelector<HTMLElement>(`[data-workflow-library-list] [data-list-row="${workflowId}"]`);

    await vi.waitFor(() => expect(row('wf-alpha')).not.toBeNull());
    expect(
      [...document.querySelectorAll<HTMLElement>('[data-workflow-library-list] [data-list-row]')].map(
        (element) => element.dataset.listRow
      )
    ).toEqual(['wf-alpha', 'wf-bundled', 'wf-user']);
    expect(cards()).toHaveLength(0);
    expect(row('wf-alpha')?.querySelector('[data-active-workflow]')?.textContent).toBe('Active');
    expect(row('wf-user')?.querySelector('[data-active-workflow]')).toBeNull();
    // Project rows never mark a template copy: the view is the project's own collection.
    expect(document.querySelector('[data-workflow-library-list] [data-workflow-in-project]')).toBeNull();

    // The active workflow is already open: no Open on its row, as its rail disables Open.
    const quickOpen = (workflowId: string) => row(workflowId)!.querySelector<HTMLElement>('[data-row-action="open"]')!;
    await act(() => userEvent.hover(row('wf-alpha')!.querySelector('[data-list-primary]')!));
    expect(getComputedStyle(quickOpen('wf-alpha')).visibility).toBe('hidden');
    await act(() => userEvent.hover(row('wf-user')!.querySelector('[data-list-primary]')!));
    expect(getComputedStyle(quickOpen('wf-user')).visibility).toBe('visible');

    // Pressing a row selects it for the rail; a double-click opens it and closes the library.
    await click(row('wf-user')!.querySelector<HTMLElement>('[data-list-primary]'));
    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-user');
    await act(() => {
      row('wf-bundled')
        ?.querySelector('[data-list-primary]')
        ?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    });
    expect(COMMANDS.selectWorkflow).toHaveBeenCalledWith('wf-bundled');
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('starts a fresh workflow from the toolbar', async () => {
    await renderDialog();

    await clickText('New workflow');

    expect(COMMANDS.createWorkflow).toHaveBeenCalledTimes(1);
    expect(onOpenChange).not.toHaveBeenCalled();
  });

  it('searches, filters by tag and orders the project’s workflows locally', async () => {
    project.setSnapshot(
      projectSnapshot([
        { document: { ...ALPHA.document, tags: 'portrait' } },
        { ...BUNDLED_COPY, document: { ...BUNDLED_COPY.document, tags: 'Portrait, upscale' } },
        USER_COPY,
      ])
    );
    await renderDialog();
    const ids = () => cards().map((element) => element.dataset.workflowCard);
    const search = document.querySelector<HTMLInputElement>('input[type="search"]')!;
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

    // Tags come from the project's own workflows, case variants merged.
    expect(
      [...document.querySelectorAll<HTMLElement>('[data-library-tag]')].map((tag) => tag.dataset.libraryTag)
    ).toEqual(['portrait', 'upscale']);

    await act(async () => {
      setValue?.call(search, 'copy');
      search.dispatchEvent(new Event('input', { bubbles: true }));
      await settleFrame();
    });
    // No debounce and no library request: the collection is local.
    expect(ids()).toEqual(['wf-bundled', 'wf-user']);
    expect(document.querySelector('[data-workflow-library-results]')?.textContent).toBe('2 workflows');

    await click(document.querySelector<HTMLElement>('[data-library-tag="portrait"]'));
    expect(ids()).toEqual(['wf-bundled']);

    await act(async () => {
      setValue?.call(search, '');
      search.dispatchEvent(new Event('input', { bubbles: true }));
      await settleFrame();
    });
    expect(ids()).toEqual(['wf-alpha', 'wf-bundled']);

    const sortButton = document.querySelector<HTMLButtonElement>('[data-workflow-library-sort]')!;
    expect(sortButton.getAttribute('aria-label')).toBe('Sort: Project order, First to last');
    await click(sortButton);
    await vi.waitFor(() => expect(sortButton.getAttribute('aria-expanded')).toBe('true'));
    await click(
      [...document.querySelectorAll<HTMLElement>('[role="menuitemradio"]')].find(
        (item) => item.textContent === 'Last to first'
      )
    );
    expect(ids()).toEqual(['wf-bundled', 'wf-alpha']);
    expect(workflowUiStore.getSnapshot().libraryProjectOrder).toEqual({ direction: 'desc', sort: 'order' });
    expect(browse.setWorkflowLibraryBrowseFilter).not.toHaveBeenCalled();
  });

  it('clears the search and tags on close but reopens on the same place', async () => {
    project.setSnapshot(projectSnapshot([{ document: { ...ALPHA.document, tags: 'portrait' } }, USER_COPY]));
    await renderDialog({ storeBound: true });
    const search = () => document.querySelector<HTMLInputElement>('input[type="search"]')!;
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

    await act(async () => {
      setValue?.call(search(), 'alp');
      search().dispatchEvent(new Event('input', { bubbles: true }));
      await settleFrame();
    });
    await click(document.querySelector<HTMLElement>('[data-library-tag="portrait"]'));
    expect(cards()).toHaveLength(1);
    expect(document.querySelector<HTMLInputElement>('[data-library-tag] input')?.checked).toBe(true);

    await click(document.querySelector<HTMLElement>('button[aria-label="Close"]'));
    expect(workflowUiStore.getSnapshot()).toMatchObject({
      isLibraryOpen: false,
      libraryQuery: { search: '', tags: [] },
      libraryTab: 'project',
    });

    await act(async () => {
      setWorkflowLibraryOpen(true);
      await settleFrame();
    });

    expect(search().value).toBe('');
    expect(document.querySelector<HTMLInputElement>('[data-library-tag] input')?.checked).toBe(false);
    expect(cards()).toHaveLength(2);
    expect(document.querySelector('[data-library-tab="project"]')).not.toBeNull();
  });

  it('opens a workflow from a double-click or the rail, then closes', async () => {
    await renderDialog();

    await act(() => card('wf-user')?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));

    expect(COMMANDS.selectWorkflow).toHaveBeenCalledWith('wf-user');
    expect(onOpenChange).toHaveBeenCalledWith(false);

    await selectCard('wf-bundled');
    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-bundled');

    // The rail's Open, not a card's pointer-only quick action.
    await click([...(rail()?.querySelectorAll('button') ?? [])].find((button) => button.textContent === 'Open'));

    expect(COMMANDS.selectWorkflow).toHaveBeenLastCalledWith('wf-bundled');
    expect(onOpenChange).toHaveBeenCalledTimes(2);
  });

  it("opens a project tile's actions from its menu button, including rename", async () => {
    const previous = { height: window.innerHeight, width: window.innerWidth };
    await page.viewport(1280, 800);
    onTestFinished(() => page.viewport(previous.width, previous.height));
    await renderDialog();

    const button = await vi.waitFor(() => {
      const element = document.querySelector<HTMLButtonElement>('[data-workflow-card-menu="wf-bundled"]');
      expect(element).not.toBeNull();
      return element!;
    });
    // The menu button takes the pointer once the card shows it, as for a user moving onto the card first.
    await act(() => userEvent.hover(card('wf-bundled')!));
    await act(() => userEvent.click(button));

    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu] [data-menu-item="rename"]')).not.toBeNull()
    );
    // The menu is the tile's; the rail keeps the active workflow.
    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-alpha');
  });

  it("duplicates another workflow from its card's menu and selects the copy", async () => {
    await renderDialog();
    await selectCard('wf-bundled');

    await act(() => userEvent.click(card('wf-user')!, { button: 'right' }));
    const contextMenu = await vi.waitFor(() => {
      const element = document.querySelector<HTMLElement>('[data-workflow-context-menu][data-state="open"]');
      expect(element?.id).toBe('workflow-library-card-wf-user-actions');
      return element!;
    });
    await click(contextMenu.querySelector<HTMLElement>('[data-menu-item="duplicate"]'));

    expect(COMMANDS.duplicateWorkflow).toHaveBeenCalledExactlyOnceWith('wf-user', 'User copy copy');
    expect(workflowUiStore.getSnapshot().librarySelection).toEqual({ projectId: PROJECT_ID, workflowId: 'wf-copy' });
  });

  it("renames and removes another workflow from its card's menu, keeping the selected one", async () => {
    await renderDialog();
    await selectCard('wf-bundled');
    const contextMenu = () => document.querySelector<HTMLElement>('[data-workflow-context-menu][data-state="open"]');
    const raiseOnUserCopy = async () => {
      await act(() => userEvent.click(card('wf-user')!, { button: 'right' }));
      await vi.waitFor(() => expect(contextMenu()?.id).toBe('workflow-library-card-wf-user-actions'));
      expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-bundled');
    };

    await raiseOnUserCopy();
    await click(contextMenu()?.querySelector<HTMLElement>('[data-menu-item="rename"]'));
    const input = await vi.waitFor(() => {
      const element = document.querySelector<HTMLInputElement>('input[name="renameValue"]');
      expect(element?.value).toBe('User copy');
      return element!;
    });
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    await act(async () => {
      setValue?.call(input, 'User copy, tuned');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    await click(input.closest('form')?.querySelector<HTMLButtonElement>('button[type="submit"]'));
    expect(COMMANDS.renameWorkflow).toHaveBeenCalledExactlyOnceWith('wf-user', 'User copy, tuned');

    await vi.waitFor(() => expect(alertDialog()).toBeNull());
    await raiseOnUserCopy();
    await click(contextMenu()?.querySelector<HTMLElement>('[data-menu-item="remove"]'));
    expect(alertDialog()?.textContent).toContain('Remove "User copy" from this project?');
    await click(
      [...(alertDialog()?.querySelectorAll('button') ?? [])].find(
        (candidate) => (candidate.textContent ?? '').trim() === 'Remove'
      )
    );

    expect(COMMANDS.removeWorkflow).toHaveBeenCalledExactlyOnceWith('wf-user');
    expect(workflowUiStore.getSnapshot().librarySelection).toEqual({ projectId: PROJECT_ID, workflowId: 'wf-bundled' });
    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-bundled');
  });

  it('hands focus to the rail menu it opens, so the pointer can move onto it without closing it', async () => {
    await renderDialog();

    await click(railMenuTrigger());

    // Inside the dialog a menu that never took focus closes on the first pointer move onto it.
    await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull(), { timeout: 2000 });
  });

  it('shows the active workflow in the rail with Open disabled, saying why, since it is already open', async () => {
    await renderDialog();

    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-alpha');
    const open = rail()?.querySelector<HTMLButtonElement>('[data-project-workflow-open]');
    expect(open?.textContent).toBe('Open');
    expect(open?.disabled).toBe(true);
    expect(document.getElementById(open?.getAttribute('aria-describedby') ?? '')?.textContent).toBe('Active workflow');
    expect(rail()?.querySelector('[data-workflow-source]')?.textContent).toBe('Not linked to a library template');
  });

  it('offers the selected project workflow as a sheet in a narrow window, its Open disabled while active', async () => {
    const previous = { height: window.innerHeight, width: window.innerWidth };
    await page.viewport(900, 700);
    onTestFinished(() => page.viewport(previous.width, previous.height));
    await renderDialog();

    const summary = rail()!.querySelector<HTMLElement>('[data-detail-sheet-summary]')!;
    const open = summary.querySelector<HTMLButtonElement>('[data-detail-sheet-open]')!;
    expect(summary.getClientRects().length).toBeGreaterThan(0);
    // The wide rail's own Open leaves the layout; the summary's says why it is disabled.
    expect(rail()!.querySelector<HTMLElement>('[data-project-workflow-open]')!.getClientRects()).toHaveLength(0);
    expect(open.disabled).toBe(true);
    expect(document.getElementById(open.getAttribute('aria-describedby') ?? '')?.textContent).toContain(
      'Active workflow'
    );

    await selectCard('wf-bundled');
    expect(open.disabled).toBe(false);
    expect(open.hasAttribute('aria-describedby')).toBe(false);
    await click(open);
    expect(COMMANDS.selectWorkflow).toHaveBeenCalledWith('wf-bundled');
  });

  it('renames through the dialog', async () => {
    await renderDialog();
    await selectCard('wf-bundled');

    await clickRailMenuItem('rename');

    const input = document.querySelector<HTMLInputElement>('input[name="renameValue"]');
    expect(input?.value).toBe('Bundled copy');

    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    await act(async () => {
      setValue?.call(input, 'Bundled, tuned');
      input?.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    await click(input?.closest('form')?.querySelector<HTMLButtonElement>('button[type="submit"]'));

    expect(COMMANDS.renameWorkflow).toHaveBeenCalledWith('wf-bundled', 'Bundled, tuned');
  });

  it('duplicates beside the original under a copy name', async () => {
    await renderDialog();
    await selectCard('wf-bundled');

    await clickRailMenuItem('duplicate');

    expect(COMMANDS.duplicateWorkflow).toHaveBeenCalledWith('wf-bundled', 'Bundled copy copy');
    // The copy becomes the selection so the rail shows what was just made.
    expect(workflowUiStore.getSnapshot().librarySelection).toEqual({ projectId: PROJECT_ID, workflowId: 'wf-copy' });
  });

  it('removes only after the confirmation is accepted', async () => {
    await renderDialog();
    await selectCard('wf-bundled');

    await clickRailMenuItem('remove');

    expect(alertDialog()?.textContent).toContain('Remove workflow');
    expect(alertDialog()?.textContent).toContain('Remove "Bundled copy" from this project?');
    expect(COMMANDS.removeWorkflow).not.toHaveBeenCalled();

    const confirm = [...(alertDialog()?.querySelectorAll('button') ?? [])].find(
      (candidate) => (candidate.textContent ?? '').trim() === 'Remove'
    );
    await click(confirm);

    expect(COMMANDS.removeWorkflow).toHaveBeenCalledWith('wf-bundled');
    // Removing the selected workflow clears the selection.
    expect(workflowUiStore.getSnapshot().librarySelection).toBeNull();
  });

  it('offers a template update only for a source the account can write', async () => {
    await renderDialog();
    await selectCard('wf-bundled');
    await openRailMenu();

    expect(rail()?.querySelector('[data-workflow-source]')?.textContent).toBe('From a bundled template (read-only)');
    expect(menuItem('save-to-library')).not.toBeNull();
    expect(menuItem('update-template')).toBeNull();

    await closeRailMenu();
    await selectCard('wf-user');
    await openRailMenu();

    expect(rail()?.querySelector('[data-workflow-source]')?.textContent).toBe(
      'Linked to library template at revision 4'
    );
    expect(menuItem('update-template')).not.toBeNull();

    await click(menuItem('update-template'));

    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({ kind: 'update-source', workflowId: 'wf-user' });
  });

  it('says when the linked template is newer, and updates the copy from it after confirmation', async () => {
    templateRecords.set('wf-portrait', { name: 'Portrait Studio', revision: 6 });
    await renderDialog();
    await selectCard('wf-user');

    const linked = () => rail()?.querySelector<HTMLElement>('[data-linked-template="wf-portrait"]');
    await vi.waitFor(() =>
      expect(linked()?.querySelector('[data-template-newer]')?.textContent).toBe('Template is newer')
    );
    // The card names the template once its record arrives; the sheet's summary says it is newer too.
    expect(linked()?.querySelector('[data-workflow-source]')?.textContent).toBe(
      'Linked to Portrait Studio, revision 4'
    );
    expect(linked()?.querySelector('[data-template-revision]')?.textContent).toBe('Library template is at revision 6');
    expect(rail()?.querySelector('[data-detail-sheet-summary]')?.textContent).toContain('Template is newer');

    const linkedButton = (text: string) =>
      [...(linked()?.querySelectorAll('button') ?? [])].find((button) => button.textContent === text);
    await click(linkedButton('Update from template'));
    expect(alertDialog()?.textContent).toContain('Replace "User copy"');
    expect(loader.replace).not.toHaveBeenCalled();
    await click(
      [...(alertDialog()?.querySelectorAll('button') ?? [])].find((button) => button.textContent === 'Update')
    );
    expect(loader.replace).toHaveBeenCalledWith(
      { name: 'Portrait Studio', revision: 6, workflow_id: 'wf-portrait' },
      'wf-user'
    );

    // Updating the library template writes this copy over the template it is linked to, named as in the menu.
    await click(linkedButton('Update library template'));
    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({ kind: 'update-source', workflowId: 'wf-user' });
  });

  it('keeps the linked template card quiet when the copy is current or the template is bundled', async () => {
    templateRecords.set('wf-portrait', { name: 'Portrait Studio', revision: 4 });
    await renderDialog();
    await selectCard('wf-user');

    await vi.waitFor(() => expect(rail()?.querySelector('[data-template-revision="4"]')).not.toBeNull());
    expect(rail()?.querySelector('[data-template-newer]')).toBeNull();

    // A bundled template is never read for its revision, and only its update is offered.
    await selectCard('wf-bundled');
    const linked = rail()?.querySelector<HTMLElement>('[data-linked-template="default_text_to_image"]');
    expect(linked?.querySelector('[data-template-revision]')).toBeNull();
    const buttons = [...(linked?.querySelectorAll('button') ?? [])].map((button) => button.textContent);
    expect(buttons).toEqual(['Update from template']);

    // An unlinked workflow has nothing to update from or save to.
    await selectCard('wf-alpha');
    expect(rail()?.querySelector('[data-linked-template="none"]')?.querySelectorAll('button')).toHaveLength(0);
  });

  it('leads a project card with its last run and lists the run and edit times in the rail', async () => {
    const completedAt = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
    project.setSnapshot(
      projectSnapshot([
        { ...ALPHA, lastRun: { completedAt, imageName: 'missing.png', submittedAt: completedAt } },
        USER_COPY,
      ])
    );
    await renderDialog();

    await vi.waitFor(() => expect(card('wf-alpha')?.textContent).toContain('Ran 2 hours ago'));
    expect(card('wf-user')?.textContent).toContain('1 node');
    expect(rail()?.textContent).toContain('Active workflow · ran 2 hours ago');
    const facts = rail()?.querySelector<HTMLElement>('[data-project-workflow-facts]');
    expect(facts?.textContent).toContain('Last run2 hours ago');
    expect(facts?.textContent).toContain('Edited');
  });

  it('reopens on the active workflow, so its template update writes the workflow being edited', async () => {
    const LANDSCAPE_COPY: ProjectWorkflowEntry = {
      document: { ...createProjectGraph('wf-landscape'), name: 'Landscape copy' },
      source: { libraryWorkflowId: 'wf-landscape-template', revision: 2 },
    };
    const workflows = [...PROJECT_WORKFLOWS, LANDSCAPE_COPY];

    project.setSnapshot(projectSnapshot(workflows, 'wf-user'));
    openWorkflowLibraryAtProjectWorkflow(PROJECT_ID, 'wf-user');
    await renderDialog({ storeBound: true });
    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-user');

    // The library closes, another template's copy becomes the workflow being edited, and Library… opens it again.
    await act(async () => {
      setWorkflowLibraryOpen(false);
      project.setSnapshot(projectSnapshot(workflows, 'wf-landscape'));
      await settleFrame();
    });
    await act(async () => {
      setWorkflowLibraryOpen(true);
      await settleFrame();
    });

    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-landscape');

    await clickRailMenuItem('update-template');

    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({
      kind: 'update-source',
      workflowId: 'wf-landscape',
    });
  });

  it('keeps a public workflow copy read-only for Update Template when another account opens the project', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-b' } };
    project.setSnapshot(projectSnapshot(PROJECT_WORKFLOWS, 'wf-user'));
    openWorkflowLibraryAtProjectWorkflow(PROJECT_ID, 'wf-user');
    await renderDialog({ storeBound: true });

    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-user');
    await openRailMenu();

    expect(menuItem('update-template')).toBeNull();
    expect(menuItem('save-to-library')).not.toBeNull();
  });

  it('previews a project workflow from its own document, from the rail or its card', async () => {
    await renderDialog();
    // Pointer-only quick actions mount on the card's first pointer visit.
    await vi.waitFor(() => expect(card('wf-user')).not.toBeNull());
    await act(() => userEvent.hover(card('wf-user')!));
    const quickPreview = card('wf-user')
      ?.closest('[data-workflow-card-root]')
      ?.querySelector<HTMLButtonElement>('[data-card-action="preview"]');
    expect(quickPreview).not.toBeNull();
    await click(quickPreview);
    await act(async () => {
      await vi.waitFor(() => expect(document.querySelector('[data-preview-dialog]')).not.toBeNull(), {
        timeout: 2000,
      });
    });
    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-graph-id')).toBe('wf-user');
    await clickText('Close preview');
    await clickText('Finish preview exit');

    await selectCard('wf-user');

    await clickText('Preview graph');
    await act(async () => {
      await vi.waitFor(() => expect(document.querySelector('[data-preview-dialog]')).not.toBeNull(), {
        timeout: 2000,
      });
    });

    const preview = document.querySelector('[data-preview-dialog]');
    expect(preview?.getAttribute('data-graph-id')).toBe('wf-user');
    expect(preview?.getAttribute('data-source-label')).toBe('User copy');
    expect(preview?.textContent).toContain('integer');
  });
});
