import type { StarterModel } from '@features/models';
import type { WorkflowModelRequirement } from '@features/workflow/core/modelRequirements';
import type {
  FieldInputTemplate,
  InvocationTemplate,
  InvocationTemplatesSnapshot,
  ProjectGraphState,
  ProjectWorkflowEntry,
  WorkflowEdge,
  WorkflowInvocationNode,
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
  setWorkflowLibrarySearch,
  setWorkflowLibraryTab,
  workflowUiStore,
} from '@features/workflow/ui/workflowUiStore';
import {
  buildInvocationNode,
  createProjectGraph,
  projectGraphReducer,
  serializeWorkflowJson,
} from '@features/workflow/utility';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { settleAnimations } from '@platform/browser/settleAnimations.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeAll, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { LibraryCopyChoiceHost } from './LibraryCopyChoiceDialog';
import { loadGraphPreview } from './libraryPreviewSource';
import { WorkflowLibraryDialog } from './WorkflowLibraryDialog';

/** Zag selects only the highlighted item, so a menu item takes a real pointer click, which hovers it first. */
const selectMenuItem = async (item: HTMLElement | null | undefined) => {
  expect(item, 'no such menu item').toBeTruthy();
  await act(() => userEvent.click(item!));
};

const GALLERY_HOST: GalleryHost = {
  galleryValues: {},
  notifications: { add: vi.fn(), reportError: vi.fn() },
  projectName: '',
};

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

const previewField = (
  name: string,
  typeName: string,
  overrides: Partial<FieldInputTemplate> = {}
): FieldInputTemplate => ({
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required: false,
  title: name.replace(/^./, (letter) => letter.toUpperCase()).replaceAll('_', ' '),
  type: { batch: false, cardinality: 'SINGLE', name: typeName },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});
const previewTemplate = (type: string, title: string, inputs: FieldInputTemplate[]): InvocationTemplate => ({
  ...PREVIEW_NODE_TEMPLATE,
  inputs: Object.fromEntries(inputs.map((input) => [input.name, input])),
  outputType: `${type}_output`,
  title,
  type,
});

const GRAPH_TEMPLATES: Record<string, InvocationTemplate> = {
  collect: previewTemplate('collect', 'Collect', [previewField('item', 'AnyField')]),
  denoise_latents: previewTemplate('denoise_latents', 'Denoise Latents', [
    previewField('steps', 'IntegerField'),
    previewField('cfg_scale', 'FloatField'),
    previewField('positive', 'ConditioningField'),
    previewField('unet', 'UNetField'),
  ]),
  l2i: previewTemplate('l2i', 'Latents to Image', [previewField('latents', 'LatentsField')]),
  main_model_loader: previewTemplate('main_model_loader', 'Main Model', [
    previewField('model', 'ModelIdentifierField', { required: true, uiModelBase: ['flux'], uiModelType: ['main'] }),
  ]),
  string: previewTemplate('string', 'String Primitive', [previewField('value', 'StringField')]),
};

const templatesState = vi.hoisted(() => ({ current: null as unknown as InvocationTemplatesSnapshot }));
const TEMPLATES_SNAPSHOT: InvocationTemplatesSnapshot = {
  error: null,
  status: 'loaded',
  templates: { integer: PREVIEW_NODE_TEMPLATE, ...GRAPH_TEMPLATES },
};

beforeEach(() => {
  templatesState.current = TEMPLATES_SNAPSHOT;
  installs.active = new Set();
});

const identity = vi.hoisted(() => ({
  session: { multiuserEnabled: false, phase: 'ready', user: null } as {
    multiuserEnabled: boolean;
    phase: 'ready';
    user: { user_id: string } | null;
  },
}));

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationTemplatesSnapshot: () => templatesState.current,
}));

const templateRecords = vi.hoisted(() => new Map<string, { name: string; revision: number }>());

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

const downloads = vi.hoisted(() => ({ downloadText: vi.fn((_text: string, _name: string, _type: string) => {}) }));

vi.mock('@platform/browser/downloadBlob', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  downloadText: downloads.downloadText,
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuthSession: () => identity.session,
}));

const browse = vi.hoisted(() => ({
  ensureWorkflowLibraryBrowseLoaded: vi.fn((_opening: Partial<WorkflowLibraryBrowseSnapshot['filter']>) =>
    Promise.resolve()
  ),
  loadWorkflowLibraryPlaceTotals: vi.fn((_places: readonly string[]) => Promise.resolve()),
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

const installs = vi.hoisted(() => ({ active: new Set<string>() }));

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
  useActiveInstallSources: () => installs.active,
  useInstallActions: () => ({ install: vi.fn(), installMany: vi.fn(), pendingSources: new Set() }),
  useModelsSelector: (selector: (snapshot: unknown) => unknown) => selector({ models: [] }),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ response: { starter_models: [FLUX_STARTER] } }),
}));

const TRANSLATIONS: Record<string, string> = {
  'common.cancel': 'Cancel',
  'common.close': 'Close',
  'workflowLibrary.activeWorkflow': 'Active',
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
  'workflowLibrary.openWithEllipsis': 'Open…',
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
  'workflowLibrary.projectCopyRevision_one': '{{count}} copy · revision {{revision}}',
  'workflowLibrary.ranWhen': 'Ran {{when}}',
  'workflowLibrary.readyToRun': 'Ready to run',
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
  'graphPreview.nodeSummary.denoise_one': '{{count}} step · cfg {{cfg}}',
  'graphPreview.nodeSummary.denoise_other': '{{count}} steps · cfg {{cfg}}',
  'workflowLibrary.preview.action': 'Preview',
  'workflowLibrary.preview.actionHint': 'Preview',
  'workflowLibrary.preview.atFirst': 'This is the first workflow here',
  'workflowLibrary.preview.atLast': 'This is the last workflow here',
  'workflowLibrary.preview.backTo': 'Back to {{place}}',
  'workflowLibrary.preview.copyJson': 'Copy workflow JSON',
  'workflowLibrary.preview.downloadJson': 'Download workflow JSON',
  'workflowLibrary.preview.edgeLabel': '{{source}} to {{target}}',
  'workflowLibrary.preview.exportMenu': 'Export',
  'workflowLibrary.preview.find': 'Find a node',
  'workflowLibrary.preview.findMatches_one': '{{count}} match',
  'workflowLibrary.preview.findMatches_other': '{{count}} matches',
  'workflowLibrary.preview.findNone': 'No matches',
  'workflowLibrary.preview.fit': 'Fit',
  'workflowLibrary.preview.from': 'From',
  'workflowLibrary.preview.helpersFolded_one': '{{count}} helper folded',
  'workflowLibrary.preview.helpersFolded_other': '{{count}} helpers folded',
  'workflowLibrary.preview.legend': 'Markers',
  'workflowLibrary.preview.loading': 'Loading workflow…',
  'workflowLibrary.preview.minimap': 'Minimap',
  'workflowLibrary.preview.needsModel': 'Needs a model',
  'workflowLibrary.preview.needsModelInline': 'needs a model',
  'workflowLibrary.preview.needsModelNote': "Needs a model that isn't installed",
  'workflowLibrary.preview.next': 'Next workflow',
  'workflowLibrary.preview.nodePackMissing': 'Node pack missing',
  'workflowLibrary.preview.position': '{{position}} of {{total}}',
  'workflowLibrary.preview.previous': 'Previous workflow',
  'workflowLibrary.preview.region': 'Preview of {{name}}',
  'workflowLibrary.preview.result': 'Result',
  'workflowLibrary.preview.resultInline': 'result',
  'workflowLibrary.preview.showAll': 'Show all {{count}}',
  'workflowLibrary.preview.showingAll_other': 'Showing all {{count}} nodes, as drawn',
  'workflowLibrary.preview.showingNodes_one': 'Showing {{shown}} of {{count}} node',
  'workflowLibrary.preview.showingNodes_other': 'Showing {{shown}} of {{count}} nodes',
  'workflowLibrary.preview.simplify': 'Simplify',
  'workflowLibrary.preview.stepper': 'Step through workflows',
  'workflowLibrary.preview.to': 'To',
  'workflowLibrary.preview.unknownNode': "Unknown node type: its node pack isn't installed on this server.",
  'workflowLibrary.preview.unknownNodeMarker': 'unknown node type',
  'workflowLibrary.preview.unreadable': "This workflow can't be previewed: {{message}}",
  'workflowLibrary.preview.youSetThis': 'You set this',
  'workflowLibrary.preview.valueNumber': '{{value}}',
  'workflowLibrary.preview.youSetThisInline': 'you set this',
  'workflowLibrary.preview.zoomIn': 'Zoom in',
  'workflowLibrary.preview.zoomOut': 'Zoom out',
  'workflowLibrary.preview.zoomPercent': '{{percent}}%',
  'workflowLibrary.preview.authorNotes': "Author's notes",
  'workflowLibrary.preview.byAuthor': 'by {{author}}',
  'workflowLibrary.preview.emptyValue': 'Empty',
  'graphPreview.copyFailed': 'Failed to copy JSON',
  'graphPreview.copyGraphJson': 'Copy graph JSON',
  'graphPreview.invalidTitle': "This graph can't compile yet.",
  'nodes.forLoopLinkageInvalid': 'Loop linkage must connect a For to its ForReturn',
  'nodes.forLoopValidationFailed': 'For loop validation failed',
  'workflowLibrary.preview.graph': 'Graph',
  'workflowLibrary.preview.graphJson': 'Graph JSON',
  'workflowLibrary.preview.nodes': 'Nodes',
  'workflowLibrary.preview.inputCount_one': '{{count}} input',
  'workflowLibrary.preview.inputCount_other': '{{count}} inputs',
  'workflowLibrary.preview.inputs': 'Inputs',
  'workflowLibrary.preview.kindLibrary': 'My library',
  'workflowLibrary.preview.kindProject': 'In this project',
  'workflowLibrary.preview.kindShared': 'Shared',
  'workflowLibrary.preview.kindTemplate': 'Template',
  'workflowLibrary.preview.modelNotInstalled': 'Model not installed',
  'workflowLibrary.preview.modelToPick': 'Model to pick',
  'workflowLibrary.preview.more': '+{{count}}',
  'workflowLibrary.preview.noInputs': 'This workflow exposes no inputs. Its values are set on its nodes.',
  'workflowLibrary.preview.pickAfterOpening': 'Pick one after opening',
  'workflowLibrary.preview.previewNamed': 'Preview {{name}}',
  'workflowLibrary.preview.readAllNotes': 'Read all notes',
  'workflowLibrary.preview.seeAllInputs_other': 'See all {{count}} inputs',
  'workflowLibrary.preview.showLessNotes': 'Show less',
  'workflowLibrary.preview.showNode': '{{node}}: show in the graph',
  'workflowLibrary.preview.version': 'v{{version}}',
  'workflowLibrary.preview.youSet': 'You set',
  'workflowLibrary.preview.positionUnknown': "Its place in these results isn't loaded",
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

const graphNode = (
  id: string,
  type: string,
  position: { x: number; y: number },
  inputs: Record<string, unknown> = {},
  fieldLabels: Record<string, string> = {}
): WorkflowInvocationNode => ({
  data: {
    inputs: Object.fromEntries(
      Object.entries(inputs).map(([name, value]) => [name, { label: fieldLabels[name] ?? '', name, value }])
    ),
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type,
    useCache: true,
    version: '1.0.0',
  },
  id,
  position,
  type: 'invocation',
});
const graphEdge = (source: string, target: string, sourceHandle: string, targetHandle: string): WorkflowEdge => ({
  id: `${source}-${target}`,
  source,
  sourceHandle,
  target,
  targetHandle,
  type: 'default',
});
const exposedField = (id: string, nodeId: string, fieldName: string) => ({
  data: { fieldIdentifier: { fieldName, nodeId }, showDescription: false, showShuffle: false },
  id,
  parentId: 'root',
  type: 'node-field' as const,
});

/** Simplified, the prompt folds into a pill on the denoiser and the collect hides: four of six nodes show. */
const GRAPH_DOCUMENT: ProjectGraphState = {
  ...createProjectGraph('graph-fixture', 'Graph Fixture'),
  edges: [
    graphEdge('prompt', 'collect', 'value', 'item'),
    graphEdge('collect', 'denoise', 'collection', 'positive'),
    graphEdge('model', 'denoise', 'unet', 'unet'),
    graphEdge('denoise', 'decode', 'latents', 'latents'),
    graphEdge('decode', 'refine', 'image', 'image'),
  ],
  author: 'Invoke',
  form: {
    elements: {
      'field-model': exposedField('field-model', 'model', 'model'),
      'field-prompt': exposedField('field-prompt', 'prompt', 'value'),
      'field-steps': exposedField('field-steps', 'denoise', 'steps'),
      'heading-model': { data: { content: 'Model' }, id: 'heading-model', parentId: 'root', type: 'heading' },
      'heading-prompt': { data: { content: 'Prompting' }, id: 'heading-prompt', parentId: 'root', type: 'heading' },
      root: {
        data: {
          children: ['heading-prompt', 'field-prompt', 'field-steps', 'heading-model', 'field-model'],
          layout: 'column',
        },
        id: 'root',
        type: 'container',
      },
    },
    rootElementId: 'root',
  },
  notes: Array.from({ length: 12 }, (_, line) => `Note line ${line + 1}: keep the refine pass for portraits.`).join(
    '\n'
  ),
  workflowVersion: '1.2.0',
  nodes: [
    graphNode(
      'prompt',
      'string',
      { x: 0, y: 0 },
      { value: 'a lighthouse at dusk, long exposure' },
      { value: 'Prompt' }
    ),
    graphNode('collect', 'collect', { x: 320, y: 0 }),
    graphNode('model', 'main_model_loader', { x: 0, y: 240 }),
    graphNode('denoise', 'denoise_latents', { x: 640, y: 120 }, { cfg_scale: 3.5, steps: 30 }),
    graphNode('decode', 'l2i', { x: 960, y: 120 }),
    graphNode('refine', 'mask_refine', { x: 1280, y: 120 }),
  ],
};
const GRAPH_ENTRY = entry('wf-graph', 'Graph Fixture', readyEnrichment(6, 'flux', NO_REQUIREMENTS, GRAPH_DOCUMENT));

/** ~80 characters with no break opportunity the rail could exploit. */

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
const LIBRARY_PREFERENCES = {
  themeId: 'classic',
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

  const wait = (ms: number) =>
    act(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, ms);
      });
    });

  const stage = () => document.querySelector<HTMLElement>('[data-preview-stage]');
  const waitForPreviewGraph = () =>
    act(async () => {
      await vi.waitFor(
        () => {
          expect(stage()?.querySelector('.react-flow__node')).not.toBeNull();
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

  const resizeViewport = async (width: number, height: number) => {
    const previous = { height: window.innerHeight, width: window.innerWidth };

    await page.viewport(width, height);
    onTestFinished(() => page.viewport(previous.width, previous.height));
  };

  const gridViewport = () => document.querySelector<HTMLElement>('[role="region"][aria-label="Workflows"]');
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
    accountLifecycle.invalidate();
    setWorkflowLibraryTab('user');
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('renders one card per entry with its base model, readiness and leading fact', async () => {
    await openWith(withSnapshot({ entries: [...LOADED_SNAPSHOT.entries, PREVIEW_FIXTURE] }));

    expect(cards()).toHaveLength(5);

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

  it('kicks the first load when it opens and leaves the store alone while closed', async () => {
    await act(() => browse.setSnapshot?.(LOADED_SNAPSHOT));
    await renderDialog(false);

    expect(browse.ensureWorkflowLibraryBrowseLoaded).not.toHaveBeenCalled();

    await renderDialog(true);

    // Calls are not counted: StrictMode double-invokes mount effects; the store's single-flight makes that a no-op.
    expect(browse.ensureWorkflowLibraryBrowseLoaded).toHaveBeenCalled();
  });

  const sortTrigger = () => document.querySelector<HTMLButtonElement>('[data-workflow-library-sort]');
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
    expect(browse.loadWorkflowLibraryPlaceTotals).toHaveBeenCalledWith(['recent', 'default']);
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

  it('switches to the bundled defaults once when the account has no workflows of its own', async () => {
    await openWith(withSnapshot({ placeTotals: { user: 0 } }));
    await wait(0);

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ place: 'default', tags: [] });
    expect(workflowUiStore.getSnapshot().libraryTab).toBe('default');
    expect(placeButton('default')?.getAttribute('aria-pressed')).toBe('true');
  });

  it('clears the checked tags but keeps the search when the place changes', async () => {
    await openWith(LOADED_SNAPSHOT);
    await clickTag('portrait');
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    const input = document.querySelector<HTMLInputElement>('input[type="search"]')!;
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    await act(async () => {
      setValue?.call(input, 'portrait');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    browse.setWorkflowLibraryBrowseFilter.mockClear();

    await act(() => placeButton('default')!.click());

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({
      direction: 'desc',
      place: 'default',
      search: 'portrait',
      sort: 'updated',
      tags: [],
    });
    expect(input.value).toBe('portrait');
    await act(() => vi.advanceTimersByTimeAsync(1000));
    vi.useRealTimers();
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
    expect(placeButton('shared')?.getAttribute('aria-pressed')).toBe('true');
    expect(placeButton('user')?.getAttribute('aria-pressed')).toBe('false');
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith(expect.objectContaining({ place: 'shared' }));
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

    expect(document.querySelector('[role="menuitem"][data-value="rename"]')).toBeNull();
    expect(document.querySelector('[role="menuitem"][data-value="delete"]')).toBeNull();
    expect(document.querySelector('[aria-label="workflowLibrary.thumbnailChoose"]')).toBeNull();
    expect(document.querySelector('[data-thumbnail-remove]')).toBeNull();
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
    expect(document.querySelector('[role="menuitem"][data-value="rename"]')).toBeNull();
    expect(document.querySelector('[role="menuitem"][data-value="delete"]')).toBeNull();
    expect(document.querySelector('[aria-label="workflowLibrary.thumbnailChoose"]')).toBeNull();
  });

  it('keeps visibility out of single-user cards and rails, even for a public template', async () => {
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    await openWith(withSnapshot({ entries: [OWNER_PUBLIC] }));

    expect(document.querySelector('[data-workflow-visibility]')).toBeNull();
    expect(document.querySelector('[data-workflow-detail] input[type="checkbox"]')).toBeNull();
    expect(document.querySelector('[data-workflow-detail]')?.textContent).not.toContain('Visibility');
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
    await expect.poll(() => browse.setWorkflowLibraryBrowseFilter.mock.calls).toContainEqual([{ search: 'photo' }]);
    browse.setWorkflowLibraryBrowseFilter.mockClear();

    await act(async () => {
      document.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click();
      await settleFrame();
    });

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

    // React tracks input values through its own setter; assigning `.value` directly would read as no change.
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

    await expect.poll(() => browse.setWorkflowLibraryBrowseFilter.mock.calls).toEqual([[{ search: 'photo' }]]);

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

  it('auto-selects the first entry and moves the selection when it disappears', async () => {
    await openWith(LOADED_SNAPSHOT);

    expect(card('wf-portrait')?.getAttribute('aria-pressed')).toBe('true');
    expect(card('wf-landscape')?.getAttribute('aria-pressed')).toBe('false');

    await act(() => card('wf-landscape')?.click());

    expect(card('wf-landscape')?.getAttribute('aria-pressed')).toBe('true');

    await act(() => browse.setSnapshot?.(withSnapshot({ entries: [SKETCH, UPSCALE] })));

    expect(card('wf-sketch')?.getAttribute('aria-pressed')).toBe('true');
  });

  it('opens the selected workflow on double click while its metadata page is evicted', async () => {
    preferences.setSnapshot({ ...preferences.port.getSnapshot(), workflowLibraryLayout: 'list' });
    await openWith(withSnapshot({ entries: [PORTRAIT, LANDSCAPE], total: 24 }));
    const primary = () => document.getElementById('workflow-library-card-wf-landscape')!;
    await act(() => primary().click());
    await act(() =>
      browse.setSnapshot?.(
        withSnapshot({
          entries: [SKETCH],
          selectedEntry: LANDSCAPE,
          startIndex: 20,
          total: 24,
        })
      )
    );
    await act(() => primary().dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));

    expect(loader.open).toHaveBeenCalledExactlyOnceWith(LANDSCAPE.item, 'first-copy');
  });

  it('keeps the loaded grid and reports a failed load-more inline', async () => {
    await openWith(withSnapshot({ error: 'Failed to load more workflows.', status: 'error' }));

    expect(cards()).toHaveLength(4);
    expect(document.body.textContent ?? '').toContain('Failed to load more workflows.');
    expect(document.body.textContent ?? '').not.toContain('No workflows match these filters.');
  });

  it("opens a tile's actions from its own menu button, toggles it closed, and hands focus back to the button", async () => {
    await openWith(LOADED_SNAPSHOT);

    const button = document.querySelector<HTMLButtonElement>('[data-workflow-card-menu="wf-landscape"]');
    expect(button).not.toBeNull();
    expect(document.querySelector('[data-workflow-card="wf-landscape"]')?.tagName).toBe('BUTTON');
    expect(document.querySelector('[data-workflow-card="wf-landscape"]')?.contains(button)).toBe(false);

    await act(async () => {
      button?.click();
      await settleFrame();
    });

    expect(document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail).toBe('wf-portrait');
    const open = () =>
      document.querySelector<HTMLElement>(
        '[data-workflow-context-menu][data-state="open"] [role="menuitem"][data-value="open"]'
      );
    await vi.waitFor(() => expect(open()).not.toBeNull());
    await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull(), { timeout: 2000 });

    await act(async () => {
      await userEvent.keyboard('{Escape}');
    });
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).toBeNull()
    );
    expect(document.activeElement).toBe(button);

    // Pointer sequence: the pointerdown lands outside the menu and must not dismiss it before the click toggles it.
    await pressTileMenuButton(button);
    await vi.waitFor(() => expect(open()).not.toBeNull());
    expect(button?.getAttribute('aria-expanded')).toBe('true');
    await pressTileMenuButton(button);
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).toBeNull()
    );
    expect(button?.getAttribute('aria-expanded')).toBe('false');
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

        await selectMenuItem(openMenu()?.querySelector<HTMLElement>('[role="menuitem"][data-value="rename"]'));
        await act(settleFrame);
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

      const open = () => openMenu()?.querySelector<HTMLElement>('[role="menuitem"][data-value="open"]');
      await vi.waitFor(() => expect(open()).not.toBeUndefined());
      await selectMenuItem(open());

      expect(loader.open).toHaveBeenCalledWith(LANDSCAPE.item, 'first-copy');
      expect(detail()).toBe('wf-portrait');
    });

    it('deletes another card and keeps the selection; deleting the selected one clears it', async () => {
      await openWith(LOADED_SNAPSHOT);
      await act(() => card('wf-upscale')?.click());
      await raise('right-click');

      await vi.waitFor(() =>
        expect(openMenu()?.querySelector('[role="menuitem"][data-value="delete"]')).not.toBeNull()
      );
      await selectMenuItem(openMenu()?.querySelector<HTMLElement>('[role="menuitem"][data-value="delete"]'));
      await act(settleFrame);
      await vi.waitFor(() => expect(alertDialog()?.textContent).toContain('Landscape Pass'));
      await confirmIn('Delete');

      expect(libraryWrites.delete).toHaveBeenCalledExactlyOnceWith('wf-landscape', expect.anything());
      await act(() => browse.setSnapshot?.(withSnapshot({ entries: [PORTRAIT, SKETCH, UPSCALE], total: 23 })));
      expect(card('wf-landscape')).toBeNull();
      expect(detail()).toBe('wf-upscale');

      await act(() =>
        card('wf-upscale')?.dispatchEvent(
          new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
        )
      );
      await vi.waitFor(() =>
        expect(openMenu()?.querySelector('[role="menuitem"][data-value="delete"]')).not.toBeNull()
      );
      await selectMenuItem(openMenu()?.querySelector<HTMLElement>('[role="menuitem"][data-value="delete"]'));
      await act(settleFrame);
      await vi.waitFor(() => expect(alertDialog()?.textContent).toContain('Upscale Chain'));
      await confirmIn('Delete');
      expect(libraryWrites.delete).toHaveBeenLastCalledWith('wf-upscale', expect.anything());
      await act(() => browse.setSnapshot?.(withSnapshot({ entries: [PORTRAIT, SKETCH], total: 22 })));
      expect(detail()).toBe('wf-portrait');
    });

    it('keeps a confirmation opened from a menu on its own workflow when a refresh drops that card', async () => {
      await openWith(LOADED_SNAPSHOT);
      await raise('right-click');
      await vi.waitFor(() =>
        expect(openMenu()?.querySelector('[role="menuitem"][data-value="delete"]')).not.toBeNull()
      );
      await selectMenuItem(openMenu()?.querySelector<HTMLElement>('[role="menuitem"][data-value="delete"]'));
      await act(settleFrame);
      await vi.waitFor(() => expect(alertDialog()?.textContent).toContain('Landscape Pass'));

      await act(() => browse.setSnapshot?.(withSnapshot({ entries: [PORTRAIT, SKETCH, UPSCALE], total: 23 })));
      await settleAnimations();
      expect(alertDialog()?.getAttribute('data-state')).toBe('open');
      expect(alertDialog()?.textContent).toContain('Landscape Pass');
      await confirmIn('Cancel');
      await vi.waitFor(() => expect(alertDialog()).toBeNull());

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
        [...(openMenu()?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [])].map((item) => item.dataset.value);
      const raiseOn = async (workflowId: string) => {
        await act(() =>
          card(workflowId)?.dispatchEvent(
            new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
          )
        );
        await vi.waitFor(() => expect(openMenu()?.id).toBe(`workflow-library-card-${workflowId}-actions`));
      };

      expect(detail()).toBe('wf-portrait');
      await raiseOn('wf-shared');
      expect(items()).toEqual(['open', 'duplicate', 'fork-into-project', 'download-json']);
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(openMenu()).toBeNull());

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
      const duplicateItem = () => openMenu()?.querySelector<HTMLElement>('[role="menuitem"][data-value="duplicate"]');
      const raiseOn = async (workflowId: string) => {
        await act(() =>
          card(workflowId)?.dispatchEvent(
            new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
          )
        );
        await vi.waitFor(() => expect(openMenu()?.id).toBe(`workflow-library-card-${workflowId}-actions`));
      };

      await raiseOn('wf-landscape');
      await selectMenuItem(duplicateItem());
      await act(settleFrame);
      expect(libraryWrites.cached).toHaveBeenCalledExactlyOnceWith(
        'wf-landscape',
        expect.objectContaining({ expectedRevision: 1 })
      );
      await vi.waitFor(() => expect(libraryWrites.create).toHaveBeenCalledOnce());
      expect(libraryWrites.create.mock.calls[0]![0].name).toBe('Landscape Pass copy');

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
  });

  it('opens the selected workflow from the rail, the keyboard-reachable path', async () => {
    await openWith(LOADED_SNAPSHOT);

    const railOpen = () => document.querySelector<HTMLButtonElement>('[data-workflow-detail-open]');

    expect(railOpen()?.textContent).toBe('Open…');
    await act(() => railOpen()?.click());

    expect(workflowUiStore.getSnapshot().libraryCopyChoice?.item.workflow_id).toBe(PORTRAIT.item.workflow_id);
    expect(loader.open).not.toHaveBeenCalled();

    await act(() => card('wf-upscale')?.click());
    expect(railOpen()?.textContent).toBe('Open');
    await act(() => railOpen()?.click());

    expect(loader.open).toHaveBeenCalledWith(UPSCALE.item, 'first-copy');
  });

  const quickAction = (workflowId: string, action: 'open' | 'preview') =>
    card(workflowId)
      ?.closest('[data-workflow-card-root]')
      ?.querySelector<HTMLButtonElement>(`[data-card-action="${action}"]`) ?? null;

  it('opens from the Open quick action without taking focus, asking first when the project holds a copy', async () => {
    await openWith(LOADED_SNAPSHOT);

    await act(() => userEvent.hover(card('wf-portrait')!));
    expect(quickAction('wf-portrait', 'open')?.textContent).toBe('Open…');
    await act(() => userEvent.click(quickAction('wf-portrait', 'open')!));
    expect(workflowUiStore.getSnapshot().libraryCopyChoice?.item.workflow_id).toBe('wf-portrait');
    expect(loader.open).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(document.querySelector('[data-library-copy-choice]')).not.toBeNull());
    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(document.activeElement).toBe(card('wf-portrait')));

    await act(() => userEvent.hover(card('wf-upscale')!));
    expect(quickAction('wf-upscale', 'open')?.textContent).toBe('Open');
    await act(() => userEvent.click(quickAction('wf-upscale', 'open')!));
    expect(loader.open).toHaveBeenCalledWith(UPSCALE.item, 'first-copy');
    expect(document.activeElement).toBe(card('wf-upscale'));
  });

  describe('preview mode', () => {
    const stageButton = (label: string) =>
      [...(stage()?.querySelectorAll<HTMLButtonElement>('button') ?? [])].find(
        (button) => button.getAttribute('aria-label') === label || button.textContent === label
      ) ?? null;
    const previewNodes = () => [...(stage()?.querySelectorAll<HTMLElement>('.react-flow__node') ?? [])];
    const previewNode = (id: string) =>
      stage()?.querySelector<HTMLElement>(`.react-flow__node[data-id="${id}"]`) ?? null;

    const nodeCard = () => stage()?.querySelector<HTMLElement>('[data-preview-node-card]') ?? null;
    const stepper = () => [...document.querySelectorAll<HTMLElement>('[data-preview-stepper]')].find(isShown) ?? null;
    const stepButton = (label: string) =>
      stepper()?.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`) ?? null;
    const positionText = () => stepper()?.querySelector('[data-preview-position]')?.textContent || null;
    const railId = () => document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail;
    const railPreview = () => document.querySelector<HTMLButtonElement>('[data-workflow-preview-trigger]');
    const collection = () => document.querySelector<HTMLElement>('[data-workflow-library-collection]');
    const press = (key: string) => act(() => userEvent.keyboard(key));
    const previewGraph = async (
      snapshot: WorkflowLibraryBrowseSnapshot = withSnapshot({ entries: [GRAPH_ENTRY], total: 1 })
    ) => {
      await openWith(snapshot);
      await act(() => card('wf-graph')?.click());
      await clickText('Preview');
      await waitForPreviewGraph();
    };

    // The library preloads the graph chunk as it opens; a cold import can outlast `waitFor` under load.
    beforeAll(() =>
      Promise.all([loadGraphPreview(), import('@features/workflow/ui/graph-preview/WorkflowPreviewJson')])
    );

    beforeEach(() => {
      browse.loadWorkflowLibraryRange.mockClear();
      downloads.downloadText.mockClear();
    });

    it('takes the places rail and the collection, keeps the detail rail, and opens no second dialog', async () => {
      await openWith(withSnapshot({ entries: [PORTRAIT, GRAPH_ENTRY], total: 2 }));
      await act(() => card('wf-graph')?.click());
      await clickText('Preview');

      expect(stage()?.getAttribute('aria-label')).toBe('Preview of Graph Fixture');
      expect(document.activeElement).toBe(stage());
      expect(document.querySelectorAll('[data-scope="dialog"][data-part="content"]')).toHaveLength(1);
      expect(getComputedStyle(collection()!).visibility).toBe('hidden');
      expect(getComputedStyle(document.querySelector('[data-library-rail-slot]')!).visibility).toBe('hidden');
      expect(railId()).toBe('wf-graph');
      expect(isShown(document.querySelector('[data-workflow-detail-open]'))).toBe(true);
      expect(isShown(railPreview())).toBe(false);
      await waitForPreviewGraph();
      expect(previewNodes()).toHaveLength(4);
    });

    it('has accessible library and preview surfaces', async () => {
      const wasDark = document.documentElement.classList.contains('dark');
      document.documentElement.classList.add('dark');
      onTestFinished(() => {
        document.documentElement.classList.toggle('dark', wasDark);
      });
      await openWith(withSnapshot({ entries: [PORTRAIT, GRAPH_ENTRY], total: 2 }));
      const dialog = () => document.querySelector<HTMLElement>('[role="dialog"]')!;
      expect(
        await auditAccessibility(dialog(), { rules: { 'label-content-name-mismatch': { enabled: true } } })
      ).toEqual([]);
      await act(() => card('wf-graph')!.click());
      await act(() => railPreview()!.click());
      await waitForPreviewGraph();
      expect(
        await auditAccessibility(dialog(), { rules: { 'label-content-name-mismatch': { enabled: true } } })
      ).toEqual([]);
    });

    it('selects cards with Space and previews from the named Preview button; Escape returns to the collection', async () => {
      await openWith(LOADED_SNAPSHOT);
      await act(() => card('wf-landscape')!.focus());
      await press(' ');
      expect(stage()).toBeNull();
      expect(railId()).toBe('wf-landscape');
      expect(railPreview()?.tabIndex).toBe(0);
      await act(() => railPreview()!.focus());
      await press(' ');

      expect(stage()).not.toBeNull();
      expect(railId()).toBe('wf-landscape');

      await press('{Escape}');
      expect(stage()).toBeNull();
      expect(document.activeElement).toBe(railPreview());
      expect(onOpenChange).not.toHaveBeenCalled();

      await act(() => userEvent.hover(card('wf-portrait')!));
      await act(() => userEvent.click(quickAction('wf-portrait', 'preview')!));
      expect(stage()).not.toBeNull();
      expect(railId()).toBe('wf-portrait');

      await press('{Escape}');
      expect(stage()).toBeNull();
      await press('{Escape}');
      expect(onOpenChange).toHaveBeenCalledWith(false);
    });

    it('returns to the collection as it left it: scroll position, selection and the control that opened it', async () => {
      const entries = Array.from({ length: 40 }, (_, index) => ({
        ...PORTRAIT,
        item: { ...PORTRAIT.item, workflow_id: `scroll-${index}`, name: `Scroll ${index}` },
      }));
      await openWith(withSnapshot({ entries, total: entries.length }));
      const viewport = gridViewport()!;
      await act(async () => {
        viewport.scrollTop = 600;
        viewport.dispatchEvent(new Event('scroll'));
        await settleFrame();
      });
      await act(() => card('scroll-12')!.click());
      await act(() => userEvent.click(railPreview()!));

      await act(() => stageButton('Back to My library')!.click());

      expect(stage()).toBeNull();
      expect(gridViewport()!.scrollTop).toBe(600);
      expect(railId()).toBe('scroll-12');
      expect(document.activeElement).toBe(railPreview());
    });

    it('steps through the results in order, selecting each, and stops at the ends', async () => {
      await openWith(withSnapshot({ total: 4 }));
      await clickText('Preview');

      expect(railId()).toBe('wf-portrait');
      expect(positionText()).toBe('1 of 4');
      expect(stepButton('Previous workflow')?.getAttribute('aria-disabled')).toBe('true');

      await act(() => userEvent.click(stepButton('Next workflow')!));
      expect(railId()).toBe('wf-landscape');
      expect(positionText()).toBe('2 of 4');
      expect(document.activeElement).toBe(stepButton('Next workflow'));

      await act(() => stepButton('Next workflow')!.focus());
      await press('{ArrowDown}');
      expect(railId()).toBe('wf-sketch');
      expect(document.activeElement).toBe(stepButton('Next workflow'));
      const rail = document.querySelector<HTMLElement>('[data-workflow-detail]')!;
      expect(rail.contains(stepper())).toBe(true);
      expect(stage()?.querySelector('[data-preview-stepper]')).toBeNull();
      expect(stage()?.querySelector('[role="status"]')?.textContent).toBe('Loading workflow…');

      await press('{ArrowDown}');
      expect(railId()).toBe('wf-upscale');
      expect(positionText()).toBe('4 of 4');
      expect(stage()?.querySelector('[role="alert"]')?.textContent).toBe(
        "This workflow can't be previewed: unreadable"
      );
      expect(stepButton('Next workflow')?.getAttribute('aria-disabled')).toBe('true');
      await press('{ArrowDown}');
      expect(railId()).toBe('wf-upscale');

      await press('{ArrowUp}');
      expect(railId()).toBe('wf-sketch');

      await press('{Escape}');
      expect(document.activeElement).toBe(card('wf-sketch'));
    });

    it.each(['selection', 'search', 'filter'] as const)('ignores a late step after a new %s intent', async (change) => {
      await openWith(withSnapshot({ entries: [PORTRAIT, LANDSCAPE], total: 24 }));
      await act(() => card('wf-landscape')!.click());
      await clickText('Preview');
      let arrive!: () => void;
      browse.loadWorkflowLibraryRange.mockImplementationOnce(
        () =>
          new Promise<boolean>((resolve) => {
            arrive = () => {
              browse.setSnapshot?.({ ...browse.getSnapshot!(), entries: [PORTRAIT, LANDSCAPE, SKETCH], total: 24 });
              resolve(true);
            };
          })
      );
      await act(() => stepButton('Next workflow')!.click());
      await act(() => {
        if (change === 'selection') {
          card('wf-portrait')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        } else if (change === 'search') {
          setWorkflowLibrarySearch('cat');
        } else {
          browse.setSnapshot?.({
            ...browse.getSnapshot!(),
            filter: { ...browse.getSnapshot!().filter, search: 'cat' },
          });
        }
      });
      if (change === 'selection') {
        expect(railId()).toBe('wf-portrait');
      }
      await act(async () => {
        arrive();
        await settleFrame();
      });

      expect(railId()).toBe(change === 'selection' ? 'wf-portrait' : 'wf-landscape');
    });

    it('draws the saved document simplified, with editor titles and markers, and every node on Show all', async () => {
      await previewGraph();

      expect(previewNodes().map((node) => node.getAttribute('aria-label'))).toEqual([
        'Main Model Model not installed you set this needs a model',
        'Prompt Denoise Latents 30 steps · cfg 3.5 you set this',
        'Latents to Image',
        'mask_refine Node pack missing unknown node type result',
      ]);
      expect(previewNode('denoise')?.textContent).toBe('Prompt Denoise Latents 30 steps · cfg 3.5');
      expect(previewNode('refine')?.textContent).toBe('mask_refine Node pack missing');
      expect([...stage()!.querySelectorAll('[data-preview-legend] li')].map((item) => item.textContent)).toEqual([
        'You set this',
        'Needs a model',
        'Result',
      ]);
      expect(stage()?.querySelector('[data-preview-node-count]')?.textContent).toBe(
        'Showing 4 of 6 nodes · 2 helpers folded' + 'Show all 6'
      );

      await act(() => stageButton('Show all 6')!.click());
      await vi.waitFor(() => expect(previewNodes()).toHaveLength(6));
      expect(previewNode('prompt')?.getAttribute('aria-label')).toBe('String Primitive you set this');
      expect(stage()?.querySelector('[data-preview-node-count]')?.textContent).toBe(
        'Showing all 6 nodes, as drawn' + 'Simplify'
      );

      await act(() => stageButton('Simplify')!.click());
      await vi.waitFor(() => expect(previewNodes()).toHaveLength(4));
    });

    it('exports the saved workflow document from its menu, in the editor’s own export shape', async () => {
      await previewGraph();
      const exportItem = (value: string) =>
        document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`);
      await act(() => userEvent.click(stageButton('Export')!));
      await vi.waitFor(() => expect(exportItem('download-workflow-json')).not.toBeNull());
      expect(exportItem('copy-workflow-json')).not.toBeNull();

      await selectMenuItem(exportItem('download-workflow-json'));

      expect(downloads.downloadText).toHaveBeenCalledWith(
        JSON.stringify(serializeWorkflowJson(GRAPH_DOCUMENT), null, 2),
        'Graph Fixture.json',
        'application/json'
      );
    });

    const tabs = () => [...(stage()?.querySelectorAll<HTMLElement>('[role="tab"]') ?? [])];
    const selectTab = (name: string) =>
      act(() => userEvent.click(tabs().find((tab) => tab.textContent?.startsWith(name))!));
    const previewInputs = () => [...(stage()?.querySelectorAll<HTMLElement>('[data-preview-input]') ?? [])];
    const inputRow = (label: string) =>
      previewInputs().find((input) => input.querySelector('p')?.textContent === label) ?? null;

    it('switches between Nodes, Inputs and Graph as tabs, keeping the choice until the library closes', async () => {
      await previewGraph();

      expect(tabs().map((tab) => [tab.textContent, tab.getAttribute('aria-selected')])).toEqual([
        ['Nodes', 'true'],
        ['Inputs 3', 'false'],
        ['Graph', 'false'],
      ]);
      const panel = () => stage()!.querySelector<HTMLElement>('[role="tabpanel"]')!;
      expect(panel().getAttribute('aria-labelledby')).toBe(tabs()[0]!.id);

      await selectTab('Inputs');
      expect(panel().getAttribute('aria-labelledby')).toBe(tabs()[1]!.id);
      expect(previewInputs()).toHaveLength(3);
      expect(stage()?.querySelector('[data-preview-find]')).toBeNull();

      await press('{Escape}');
      await clickText('Preview');
      expect(tabs()[1]?.getAttribute('aria-selected')).toBe('true');
      await act(async () => {
        document.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click();
        await settleFrame();
      });
      await renderDialog(false);
      await renderDialog(true);
      await clickText('Preview');
      expect(tabs()[0]?.getAttribute('aria-selected')).toBe('true');
    });

    it('shows in Graph the graph the workflow compiles to, or in words why it does not compile', async () => {
      const json = () => stage()?.querySelector('[data-preview-json] code')?.textContent ?? null;

      await previewGraph();
      await selectTab('Graph');
      await vi.waitFor(
        async () => {
          await act(settleFrame);
          expect(json()).not.toBeNull();
        },
        { timeout: 5000 }
      );

      expect(Object.keys((JSON.parse(json()!) as { nodes: Record<string, unknown> }).nodes)).toEqual(
        expect.arrayContaining(['denoise', 'decode'])
      );

      const broken = {
        ...GRAPH_DOCUMENT,
        edges: [...GRAPH_DOCUMENT.edges, graphEdge('decode', 'denoise', 'loop_linkage', 'latents')],
      };
      await act(() =>
        browse.setSnapshot?.(
          withSnapshot({
            entries: [entry('wf-graph', 'Graph Fixture', readyEnrichment(6, 'flux', NO_REQUIREMENTS, broken))],
            total: 1,
          })
        )
      );
      expect(json()).toBeNull();
      expect(stage()?.querySelector('[role="tabpanel"] [role="alert"]')?.textContent).toBe(
        "This graph can't compile yet. For loop validation failed: Loop linkage must connect a For to its ForReturn."
      );
      await selectTab('Nodes');
      await waitForPreviewGraph();
    });

    it('shows the form read-only: saved values, a model to install, and the node each input sets', async () => {
      await previewGraph();
      await selectTab('Inputs');

      expect([...stage()!.querySelectorAll('h3')].map((heading) => heading.textContent)).toEqual([
        'Prompting',
        'Model',
      ]);
      const value = (label: string) => inputRow(label)?.querySelector<HTMLElement>('[data-input-value]');
      expect(value('Prompt')?.textContent).toBe('a lighthouse at dusk, long exposure');
      expect(value('Prompt')?.hasAttribute('tabindex')).toBe(false);
      expect(value('Steps')?.textContent).toBe('30');
      expect(value('Model')?.dataset.inputValue).toBe('missing');
      expect(value('Model')?.textContent).toBe('FLUX checkpoint · Not installed');
      const warningProbe = document.createElement('span');
      warningProbe.style.color = 'var(--chakra-colors-fg\\.warning)';
      document.body.append(warningProbe);
      const warningColor = getComputedStyle(warningProbe).color;
      warningProbe.remove();
      value('Model')!.style.color = 'rgb(1, 2, 3)';
      expect(getComputedStyle(value('Model')!).borderColor).toBe(warningColor);
      value('Model')!.style.removeProperty('color');

      await act(() => userEvent.click(inputRow('Steps')!.querySelector('button')!));
      await waitForPreviewGraph();
      expect(tabs()[0]?.getAttribute('aria-selected')).toBe('true');
      expect(nodeCard()?.dataset.previewNodeCard).toBe('denoise');
      expect(document.activeElement).toBe(nodeCard());

      await selectTab('Inputs');
      await act(() => userEvent.click(inputRow('Prompt')!.querySelector('button')!));
      await waitForPreviewGraph();
      expect(nodeCard()?.dataset.previewNodeCard).toBe('prompt');
      expect(previewNodes()).toHaveLength(6);
    });

    it('reopens on the collection and on Nodes after a close that bypasses the library’s own controls', async () => {
      await act(() => browse.setSnapshot?.(LOADED_SNAPSHOT));
      setWorkflowLibraryOpen(true);
      const render = () =>
        act(async () => {
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
      await render();
      await clickText('Preview');
      await selectTab('Inputs');
      expect(stage()).not.toBeNull();

      await act(async () => {
        setWorkflowLibraryOpen(false);
        await settleFrame();
      });
      await act(async () => {
        setWorkflowLibraryOpen(true);
        await settleFrame();
      });

      expect(stage()).toBeNull();
      expect(getComputedStyle(collection()!).visibility).toBe('visible');
      await clickText('Preview');
      expect(tabs()[0]?.getAttribute('aria-selected')).toBe('true');
    });
  });

  const NARROW_WINDOWS = [[640, 400]] as const;

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
      let centre = sheet.parentElement!;
      while (!/(auto|scroll)/.test(getComputedStyle(centre).overflowY)) {
        centre = centre.parentElement!;
      }

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

      await act(() => userEvent.click(more));
      await vi.waitFor(() => expect(more.getAttribute('aria-expanded')).toBe('true'));
      const actions = document.getElementById(more.getAttribute('aria-controls') ?? '')!;
      expect(actions.querySelector('[role="menuitem"][data-value="duplicate"]')).not.toBeNull();
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
        expect(centre.scrollTop).toBeGreaterThan(centreScrollTop);
      }
      expect(getComputedStyle(summary.querySelector('p')!).whiteSpace).not.toBe('nowrap');
      expect(
        [...details.querySelectorAll('p')].some((text) => text.textContent === 'Portrait Studio' && isShown(text))
      ).toBe(false);
      expect([...details.querySelectorAll('img')].some(isShown)).toBe(false);
      expect(details.querySelector('[data-requirement-status], [data-workflow-detail-facts]')).not.toBeNull();
      expect(
        [...details.querySelectorAll('button')].some((button) => button.textContent === 'Preview' && isShown(button))
      ).toBe(true);

      await act(() => userEvent.click(more));
      await vi.waitFor(() => expect(more.getAttribute('aria-expanded')).toBe('true'));
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(more.getAttribute('aria-expanded')).toBe('false'));
      expect(toggle.getAttribute('aria-expanded')).toBe('true');

      await act(() => details.focus());
      await act(() => userEvent.keyboard('{Escape}'));
      expect(toggle.getAttribute('aria-expanded')).toBe('false');
      expect(document.activeElement).toBe(toggle);
      expect(isShown(details)).toBe(false);
      expect(getComputedStyle(grid).visibility).toBe('visible');
      expect(grid.scrollTop).toBe(scrollTop);
      expect(centre.scrollTop).toBe(centreScrollTop);
      expectContained(search, content);
      expect(onOpenChange).not.toHaveBeenCalledWith(false);

      await act(() => userEvent.keyboard('{Escape}'));
      expect(onOpenChange).toHaveBeenCalledWith(false);
    }
  );

  describe('list layout', () => {
    const row = (workflowId: string) =>
      document.getElementById(`workflow-library-card-${workflowId}`)?.closest<HTMLElement>('[data-list-row]') ?? null;
    const primary = (workflowId: string) => row(workflowId)?.querySelector<HTMLElement>('[data-list-primary]') ?? null;
    const rail = () => document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail;

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
      expect(row('wf-portrait')?.querySelector('[data-workflow-in-project]')).not.toBeNull();
      expect(row('wf-sketch')?.querySelector('[data-enrichment-placeholder]')).not.toBeNull();
      expect(
        document
          .querySelector('[data-workflow-library-list] [role="list"]')
          ?.querySelector('[aria-setsize]')
          ?.getAttribute('aria-setsize')
      ).toBe('24');
      expect(item('Small cards')).toBeUndefined();
      expect(item('Fill')).toBeUndefined();
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

  const menuItem = (value: string) => document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`);
  const railMenuTrigger = () => rail()?.querySelector<HTMLElement>('[aria-label="More actions"]');
  // The trigger's data-state belongs to its tooltip; aria-expanded is the menu's, and its items mount a frame later.
  const openRailMenu = async () => {
    const trigger = railMenuTrigger();
    expect(trigger).not.toBeNull();

    if (trigger?.getAttribute('aria-expanded') !== 'true') {
      await click(trigger);
    }

    await vi.waitFor(
      () => {
        expect(railMenuTrigger()?.getAttribute('aria-expanded')).toBe('true');
        expect(document.querySelector('[role="menuitem"]')).not.toBeNull();
      },
      { timeout: 2000 }
    );
  };
  // Escape reaches the menu only once it holds focus a frame later; closed items stay mounted, so read aria-expanded.
  const closeRailMenu = async () => {
    if (railMenuTrigger()?.getAttribute('aria-expanded') === 'true') {
      await click(railMenuTrigger());
    }

    await vi.waitFor(() => expect(railMenuTrigger()?.getAttribute('aria-expanded')).toBe('false'), { timeout: 2000 });
  };
  const clickRailMenuItem = async (value: string) => {
    await openRailMenu();
    await selectMenuItem(menuItem(value));
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

    expect(browse.ensureWorkflowLibraryBrowseLoaded).not.toHaveBeenCalled();
    expect(browse.loadWorkflowLibraryPlaceTotals).toHaveBeenCalled();
    expect(document.querySelector('[data-library-tag]')).toBeNull();
    expect(document.querySelector('[data-workflow-library-rail]')?.textContent).toContain(
      'Edits autosave here; the library changes only when you save to it.'
    );
    expect(document.querySelector('[data-workflow-card] [data-workflow-visibility]')).toBeNull();
    expect(document.body.textContent).not.toContain('Public workflow');

    const quickOpen = (workflowId: string) =>
      card(workflowId)?.closest('[data-workflow-card-root]')?.querySelector('[data-card-action="open"]') ?? null;
    await act(() => userEvent.hover(card('wf-alpha')!));
    await act(() => userEvent.hover(card('wf-user')!));
    expect(quickOpen('wf-alpha')).toBeNull();
    expect(quickOpen('wf-user')).not.toBeNull();
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

    expect(
      [...document.querySelectorAll<HTMLElement>('[data-library-tag]')].map((tag) => tag.dataset.libraryTag)
    ).toEqual(['portrait', 'upscale']);

    await act(async () => {
      setValue?.call(search, 'copy');
      search.dispatchEvent(new Event('input', { bubbles: true }));
      await settleFrame();
    });
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

  it('opens a workflow from a double-click or the rail, then closes', async () => {
    await renderDialog();

    await act(() => card('wf-user')?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));

    expect(COMMANDS.selectWorkflow).toHaveBeenCalledWith('wf-user');
    expect(onOpenChange).toHaveBeenCalledWith(false);

    await selectCard('wf-bundled');
    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-bundled');

    await click([...(rail()?.querySelectorAll('button') ?? [])].find((button) => button.textContent === 'Open'));

    expect(COMMANDS.selectWorkflow).toHaveBeenLastCalledWith('wf-bundled');
    expect(onOpenChange).toHaveBeenCalledTimes(2);
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
    await selectMenuItem(contextMenu()?.querySelector<HTMLElement>('[role="menuitem"][data-value="rename"]'));
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
    await selectMenuItem(contextMenu()?.querySelector<HTMLElement>('[role="menuitem"][data-value="remove"]'));
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

  it('shows the active workflow in the rail with Open disabled, saying why, since it is already open', async () => {
    await renderDialog();

    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-alpha');
    const open = rail()?.querySelector<HTMLButtonElement>('[data-project-workflow-open]');
    expect(open?.textContent).toBe('Open');
    expect(open?.disabled).toBe(true);
    expect(document.getElementById(open?.getAttribute('aria-describedby') ?? '')?.textContent).toBe('Active workflow');
    expect(rail()?.querySelector('[data-workflow-source]')?.textContent).toBe('Not linked to a library template');
  });

  it('duplicates beside the original under a copy name', async () => {
    await renderDialog();
    await selectCard('wf-bundled');

    await clickRailMenuItem('duplicate');

    expect(COMMANDS.duplicateWorkflow).toHaveBeenCalledWith('wf-bundled', 'Bundled copy copy');
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

    await selectMenuItem(menuItem('update-template'));

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

    await click(linkedButton('Update library template'));
    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({ kind: 'update-source', workflowId: 'wf-user' });
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

  it('draws a project workflow afresh when its document is replaced under the same id, dropping the open card', async () => {
    const stage = () => document.querySelector<HTMLElement>('[data-preview-stage]');
    const nodes = () => [...(stage()?.querySelectorAll<HTMLElement>('.react-flow__node') ?? [])];
    await renderDialog();
    await selectCard('wf-user');
    await clickText('Preview');
    await act(async () => {
      await vi.waitFor(() => expect(nodes()).toHaveLength(1), { timeout: 2000 });
    });
    await act(() => userEvent.click(nodes()[0]!));
    expect(stage()?.querySelector('[data-preview-node-card]')).not.toBeNull();

    await act(async () => {
      project.setSnapshot(
        projectSnapshot(
          PROJECT_WORKFLOWS.map((workflow) =>
            workflow === USER_COPY
              ? { ...USER_COPY, document: { ...GRAPH_DOCUMENT, id: 'wf-user', name: 'User copy' } }
              : workflow
          )
        )
      );
      await settleFrame();
    });

    await act(async () => {
      await vi.waitFor(
        () => expect(nodes().map((node) => node.dataset.id)).toEqual(['model', 'denoise', 'decode', 'refine']),
        {
          timeout: 2000,
        }
      );
    });
    expect(stage()?.querySelector('[data-preview-node-card]')).toBeNull();
    expect(stage()?.querySelector('[data-preview-node-count]')?.textContent).toContain('Showing 4 of 6 nodes');
  });
});
