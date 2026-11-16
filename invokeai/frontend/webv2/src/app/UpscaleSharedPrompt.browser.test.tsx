import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';
import type * as IdentityModule from '@features/identity';
import type { ModelConfig } from '@features/models';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
import type * as ProjectsApiModule from '@workbench/projects/api';
import type { InvokeIssueAction } from '@workbench/shell/topbar/invokeIssueActions';
import type { InvocationIssue } from '@workbench/shell/topbar/invokeIssueTargets';
import type * as OpenWidgetModule from '@workbench/useOpenWorkbenchWidget';
import type { RegisteredWidget } from '@workbench/widgetContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { PositivePromptField } from '@features/generation/components';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { promptTemplatesQueryOptions } from '@features/generation/queries';
import { GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { createDefaultUpscaleWidgetValues } from '@features/upscale';
import { UpscaleWidgetView } from '@features/upscale/ui/UpscaleWidgetView';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { captureInvocationRuntimeInputs } from '@workbench/invocationPreparation';
import {
  DEFAULT_PREFERENCES,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
} from '@workbench/settings/store';
import { runNotificationUndo } from '@workbench/shell/notifications/undoRegistry';
import { useInvokeIssueActions } from '@workbench/shell/topbar/invokeIssueActions';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { getProjectWidgetValues, getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { WorkbenchWidgetRegistryProvider } from '@workbench/WorkbenchWidgetRegistryContext';
import { act, useMemo, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { UpscaleUiAdapterProvider } from './UpscaleUiAdapter';

const model = (key: string, base: string, type: string): ModelConfig => ({
  base,
  file_size: 1,
  format: 'checkpoint',
  hash: key,
  key,
  name: key,
  path: key,
  source: key,
  source_type: 'path',
  type,
});
const models = [
  model('upscale-main', 'sdxl', 'main'),
  model('spandrel', 'any', 'spandrel_image_to_image'),
  { ...model('tile', 'sdxl', 'controlnet'), name: 'SDXL Tile ControlNet' },
];
const modelSnapshot = { models, status: 'loaded' };
const CINEMATIC: PromptTemplateRecord = {
  hasImage: false,
  id: 'cinematic',
  isDefault: false,
  isPublic: false,
  name: 'Cinematic',
  negativePrompt: '{prompt}, lowres',
  positivePrompt: '{prompt}, film grain',
  userId: 'user-1',
};
const noop = () => undefined;
const generationAdapterBase = {
  account: { currentUserId: null, multiuserEnabled: false },
  capabilities: { canManagePromptTemplates: false, canManageSharedSystemPrompts: false },
  gallery: { findImage: noop, selectedImage: null, touchImages: noop },
  models: { catalog: [], ensureLoaded: noop, openManager: noop, status: 'loaded' },
  notifications: { error: noop, info: noop, reportError: noop, undoable: noop },
  project: { activeProjectId: 'project', invocationSourceId: 'upscale', showPromptSyntaxHighlighting: false },
  promptHistory: { clear: noop, items: [], remove: noop, restore: noop },
};

/** Section open state comes from the user's preferences, as the App adapter reads it. */
const GenerationUi = ({ children }: { children: ReactNode }) => {
  const sectionsOpen = useWorkbenchPreferenceSelector((preferences) => preferences.generateSectionsOpen);
  const adapter = useMemo(
    () =>
      ({
        ...generationAdapterBase,
        sectionPreferences: {
          sectionsOpen,
          setSectionsOpen: (changes: Readonly<Record<string, boolean>>) =>
            void patchWorkbenchPreferences({ generateSectionsOpen: { ...sectionsOpen, ...changes } }),
        },
      }) as unknown as GenerationUiAdapter,
    [sectionsOpen]
  );

  return <GenerationUiProvider adapter={adapter}>{children}</GenerationUiProvider>;
};

let store: WorkbenchInternalStore;
let host: HTMLDivElement;
let root: Root;

vi.mock('react-i18next', () => {
  // A count shows after its key, so a summary's numbers can be read.
  const t = (key: string, values?: { count?: number }) =>
    values?.count === undefined ? key : `${key}:${values.count}`;
  return { useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t }) };
});
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
  getModelImageUrl: () => '',
  useModelsSelector: (selector: (snapshot: typeof modelSnapshot) => unknown) => selector(modelSnapshot),
  useOpenModelInManager: () => undefined,
}));
vi.mock('@features/models/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ModelSelect: () => null,
}));
vi.mock('@features/gallery/mediaSlot', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  GalleryMediaSlot: () => null,
}));
// The wildcard catalog is the backend's; the prompt field only reads its names.
vi.mock('@features/generation/data/wildcards', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve([]), queryKey: ['generation', 'wildcards'] }),
}));
// Expansion is the backend's: a prompt with an unbalanced group fails to expand, as the parser would refuse it.
vi.mock('@features/generation/data/promptUtilities', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  parseDynamicPrompts: ({ prompt }: { prompt: string }) =>
    Promise.resolve(
      prompt.includes('{{')
        ? { error: 'Unbalanced braces', prompts: [prompt] }
        : { error: null, prompts: ['a red fox', 'a green fox'] }
    ),
}));
// Preferences save to the server; here they stay in memory.
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ProjectsApiModule>()),
  setClientStateValue: async () => {},
}));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<typeof IdentityModule>()),
  useCapabilities: () => ({ canManageModels: false }),
}));
vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => noop }));
vi.mock('@workbench/useOpenWorkbenchWidget', async (importOriginal) => ({
  ...(await importOriginal<typeof OpenWidgetModule>()),
  useOpenWorkbenchWidget: () => noop,
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) =>
    selector(useSyncExternalStore(store.subscribe, store.getSnapshot).activeProject),
  useWidgetValuesSelector: (widgetId: string, selector: (values: Record<string, unknown>) => unknown) => {
    const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot);
    return selector(getResolvedProjectWidgetValues(snapshot.activeProject, widgetId, snapshot.scopedDefaults));
  },
  useOptionalWorkbenchCommands: () => store.commands,
  useWorkbenchCommands: () => store.commands,
  useWorkbenchQueries: () => store.queries,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const generateValues = () => getProjectWidgetValues(store.getSnapshot().activeProject, 'generate');
const button = (label: string) => document.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);

/** The real registry, minus chunk preloading: revealing a widget changes the layout and loads nothing. */
const withoutPreload = (widget: RegisteredWidget): RegisteredWidget => ({
  ...widget,
  implementation: { ...widget.implementation, preload: () => {} },
});
const getQuietWidgetById = (typeId: string) => {
  const widget = getWidgetById(typeId);

  return widget ? withoutPreload(widget) : undefined;
};
const getQuietWidgetsForRegion = (region: WidgetRegion) => getWidgetsForRegion(region).map(withoutPreload);
const NO_LORAS: [] = [];

let getIssueActions: ((issue: InvocationIssue) => InvokeIssueAction[]) | null = null;
const publishIssueActions = (actions: (issue: InvocationIssue) => InvokeIssueAction[]) => {
  getIssueActions = actions;
};
/** Publishes the Invoke issue actions as the Invoke popover reads them. */
const IssueActionsProbe = ({
  onActions,
}: {
  onActions: (actions: (issue: InvocationIssue) => InvokeIssueAction[]) => void;
}) => {
  onActions(useInvokeIssueActions());

  return null;
};

const detailGuidanceTrigger = () =>
  [...host.querySelectorAll<HTMLButtonElement>('button')].find((candidate) =>
    candidate.textContent?.includes('widgets.upscale.detailGuidance')
  )!;
const upscalePrompt = () =>
  host.querySelector<HTMLTextAreaElement>(
    '[data-testid="upscale"] textarea[aria-label="widgets.generate.positivePrompt"]'
  )!;

const render = async ({
  openDetailGuidance = true,
  withGenerateField = false,
}: { openDetailGuidance?: boolean; withGenerateField?: boolean } = {}) => {
  const queryClient = new QueryClient();
  // Fresh for the query's stale time, so the catalog the panel lists is this one and no request leaves the test.
  queryClient.setQueryData(promptTemplatesQueryOptions().queryKey, [CINEMATIC]);
  store.commands.widgets.patchValues('upscale', {
    ...createDefaultUpscaleWidgetValues(models),
    inputImage: { height: 100, image_name: 'input.png', width: 200 },
  });
  store.commands.generation.patchPromptDraft(
    { negativePrompt: 'blur', positivePrompt: 'fine detail' },
    'upscale',
    store.getSnapshot().activeProject.id
  );

  await act(() =>
    root.render(
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>
          <WorkbenchWidgetRegistryProvider
            getWidgetById={getQuietWidgetById}
            getWidgetsForRegion={getQuietWidgetsForRegion}
          >
            <GenerationUi>
              <UpscaleUiAdapterProvider>
                <DndContext>
                  <div data-testid="upscale">
                    <UpscaleWidgetView />
                  </div>
                  {/* Generate's own prompt field, mounted after Upscale's. */}
                  {withGenerateField ? (
                    <div data-testid="generate">
                      <PositivePromptField
                        heightPx={96}
                        loras={NO_LORAS}
                        projectId="project"
                        selectedModel={undefined}
                        showSyntaxHighlighting={false}
                        value="a generate prompt"
                        onChange={noop}
                        onResizeEnd={noop}
                        onUsePrompt={noop}
                      />
                    </div>
                  ) : null}
                  <IssueActionsProbe onActions={publishIssueActions} />
                </DndContext>
              </UpscaleUiAdapterProvider>
            </GenerationUi>
          </WorkbenchWidgetRegistryProvider>
        </ChakraProvider>
      </QueryClientProvider>
    )
  );
  // The prompts live in the collapsed detail guidance section.
  if (openDetailGuidance) {
    await act(async () => {
      await userEvent.click(detailGuidanceTrigger());
    });
  }
};

/** Queues the Upscale as Invoke does once nothing is left to expand. */
const queueUpscale = () => {
  act(() =>
    store.commands.generation.submitResolved({
      backendSupportsCancellation: true,
      models,
      projectId: store.getSnapshot().activeProject.id,
      route: { destination: 'gallery', destinationLocked: false, sourceId: 'upscale', sourceLocked: false },
      runtime: captureInvocationRuntimeInputs('upscale'),
    })
  );

  return store.getSnapshot().activeProject.queue.items[0]?.snapshot.backendSubmission;
};

beforeEach(async () => {
  accountLifecycle.activate('upscale-shared-prompt-test');
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  // The account lifecycle owns the capability table, so it is seeded once the account is active.
  setArchitectureCapabilities(architectureCapabilitiesFixture);
  store = createWorkbenchStore();
  host = document.createElement('div');
  host.style.width = '420px';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  getIssueActions = null;
  resetArchitectureCapabilities();
  accountLifecycle.invalidate();
});

describe("Upscale's prompt templates and dynamic prompts", () => {
  it('applies a template to the prompt it shares with Generate, queues the prompt inside it, and undoes', async () => {
    await render();

    const templates = button('widgets.generate.promptTemplates.title');

    expect(templates).not.toBeNull();

    await act(async () => {
      await userEvent.click(templates!);
    });
    await act(async () => {
      await userEvent.click(
        [...document.querySelectorAll<HTMLButtonElement>('[data-scope="popover"][data-part="content"] button')].find(
          (candidate) => candidate.textContent?.includes('Cinematic')
        )!
      );
    });

    expect(generateValues().promptTemplate).toEqual({
      id: 'cinematic',
      name: 'Cinematic',
      negativePrompt: '{prompt}, lowres',
      positivePrompt: '{prompt}, film grain',
    });
    // The trigger names what is applied.
    expect(button('widgets.generate.promptTemplates.title')?.textContent).toContain('Cinematic');
    expect(queueUpscale()).toMatchObject({
      negativePrompt: 'blur, lowres',
      positivePrompt: 'fine detail, film grain',
    });

    const undo = store
      .getSnapshot()
      .notifications.flatMap((notification) => notification.actions ?? [])
      .find((action) => action.kind === 'undo') as { undoId: string } | undefined;

    expect(undo).toBeDefined();
    act(() => runNotificationUndo(undo!.undoId));

    expect(generateValues().promptTemplate).toBeNull();
  });

  it('shows the merged prompt read-only and goes back to the prompt as written', async () => {
    await render();
    act(() =>
      store.commands.generation.patchPromptDraft(
        { promptTemplate: { ...CINEMATIC }, promptTemplateViewMode: false },
        'upscale',
        store.getSnapshot().activeProject.id
      )
    );

    await act(async () => {
      await userEvent.click(button('widgets.generate.promptTemplates.viewMerged')!);
    });

    const prompt = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="widgets.generate.positivePrompt"]')!;

    expect(generateValues().promptTemplateViewMode).toBe(true);
    expect(prompt.readOnly).toBe(true);
    expect(prompt.value).toBe('fine detail, film grain');

    await act(async () => {
      await userEvent.click(button('widgets.generate.promptTemplates.editAuthored')!);
    });

    expect(prompt.readOnly).toBe(false);
    expect(prompt.value).toBe('fine detail');
  });

  it('turns dynamic prompts off for the shared prompt, which then queues as typed', async () => {
    await render();
    act(() =>
      store.commands.generation.patchPromptDraft(
        { positivePrompt: 'a {red|green} fox' },
        'upscale',
        store.getSnapshot().activeProject.id
      )
    );

    await act(async () => {
      await userEvent.click(button('widgets.generate.dynamicPrompts.showPrompts')!);
    });
    await act(async () => {
      await userEvent.click(
        document.querySelector<HTMLElement>(
          '[data-scope="popover"][data-part="content"] [data-scope="switch"][data-part="root"]'
        )!
      );
    });

    expect(generateValues().dynamicPromptsEnabled).toBe(false);
    expect(button('widgets.generate.dynamicPrompts.offLabel')).not.toBeNull();

    const submission = queueUpscale();

    expect(submission).toMatchObject({ positivePrompt: 'a {red|green} fox' });
    expect(submission).not.toHaveProperty('positivePrompts');
  });

  it("shows an Upscale prompt that cannot expand: opens Detail guidance, focuses Upscale's field and its note", async () => {
    await render({ openDetailGuidance: false, withGenerateField: true });
    act(() =>
      store.commands.generation.patchPromptDraft(
        { positivePrompt: 'a {{red|green} fox' },
        'upscale',
        store.getSnapshot().activeProject.id
      )
    );
    const generatePrompt = host.querySelector<HTMLTextAreaElement>('[data-testid="generate"] textarea')!;

    expect(detailGuidanceTrigger().getAttribute('aria-expanded')).toBe('false');
    expect(upscalePrompt().checkVisibility()).toBe(false);

    const show = getIssueActions!({
      message: 'The prompt could not be expanded.',
      target: { kind: 'prompt', widgetId: 'upscale' },
    }).find((action) => action.id === 'show')!;
    await act(() => show.run());

    await expect.poll(() => document.activeElement).toBe(upscalePrompt());
    expect(document.activeElement).not.toBe(generatePrompt);
    expect(detailGuidanceTrigger().getAttribute('aria-expanded')).toBe('true');
    // The note under the field says why the prompt cannot expand and offers the way out.
    await expect
      .poll(
        () =>
          [...host.querySelectorAll<HTMLButtonElement>('[data-testid="upscale"] button')]
            .find((candidate) => candidate.textContent === 'widgets.generate.dynamicPrompts.turnOff')
            ?.checkVisibility() ?? false,
        { timeout: 3000 }
      )
      .toBe(true);
  });
});

describe('Upscale Detail guidance, folded', () => {
  const summary = () => detailGuidanceTrigger().querySelector('[data-section-summary]')?.textContent ?? null;
  const patchPrompt = (patch: Parameters<WorkbenchInternalStore['commands']['generation']['patchPromptDraft']>[0]) =>
    act(() => store.commands.generation.patchPromptDraft(patch, 'upscale', store.getSnapshot().activeProject.id));

  it('names the prompt template Generate wraps around the shared prompt', async () => {
    await render({ openDetailGuidance: false });
    patchPrompt({ promptTemplate: { ...CINEMATIC } });

    expect(detailGuidanceTrigger().getAttribute('aria-expanded')).toBe('false');
    expect(summary()).toBe('Cinematic');
    // The shared badge stays beside it.
    expect(detailGuidanceTrigger().textContent).toContain('widgets.upscale.shared');
  });

  it('counts the prompts dynamic prompts expand the shared prompt to', async () => {
    await render({ openDetailGuidance: false });
    patchPrompt({ positivePrompt: 'a {red|green} fox' });

    await expect.poll(summary, { timeout: 3000 }).toBe('topbar.invoke.summary.promptCount:2');
  });

  it('says nothing for a plain prompt without a template', async () => {
    await render({ openDetailGuidance: false });

    expect(summary()).toBeNull();
  });
});
