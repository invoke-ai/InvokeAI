import type { GalleryUiAdapter } from '@features/gallery/react';
import type { Project } from '@workbench/projectContracts';
import type * as OpenWorkbenchWidgetModule from '@workbench/useOpenWorkbenchWidget';
import type { OpenWorkbenchWidgetOptions, RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';
import type { ReactNode } from 'react';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { previewWidgetManifest } from '@workbench/widgets/preview/manifest';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GalleryUiAdapterProvider } from './GalleryUiAdapter';

let store: ReturnType<typeof createWorkbenchStore>;
let adapter: GalleryUiAdapter;
const noop = () => undefined;
const livePreviewFollow = vi.fn();
const previewWidget = {
  implementation: { preload: noop },
  manifest: previewWidgetManifest,
  status: 'enabled',
} as unknown as RegisteredWidget;

vi.mock('@features/gallery/react', () => ({
  GalleryUiProvider: ({ adapter: next, children }: { adapter: GalleryUiAdapter; children: ReactNode }) => {
    adapter = next;
    return children;
  },
}));
vi.mock('@workbench/widgets/preview/livePreviewFollow', () => ({
  useLivePreviewFollow: () => ({
    sessions: [],
    gallerySessions: [],
    pinnedSessionId: null,
    followedSessionId: 'run:1',
    follow: livePreviewFollow,
    pin: vi.fn(),
    showAll: vi.fn(),
  }),
}));
vi.mock('@workbench/projects/useProjectFileActions', () => ({ useExportLibraryProject: () => noop }));
// The real open command against the test store, with a registry that holds only Preview.
vi.mock('@workbench/useOpenWorkbenchWidget', async (importOriginal) => {
  const actual = await importOriginal<typeof OpenWorkbenchWidgetModule>();
  return {
    useOpenWorkbenchWidget: () => (widgetId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions) =>
      actual.openWorkbenchWidget(store.commands.widgets, () => [previewWidget], widgetId, options),
  };
});
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(store.getSnapshot().activeProject),
  useWorkbenchCommands: () => store.commands,
  useWorkbenchInternalStore: () => store,
  useWorkbenchPersistenceService: () => ({ ensureProjectOnServer: () => Promise.resolve() }),
  useWorkbenchQueries: () => store.queries,
}));

const renderAdapter = () => {
  renderToStaticMarkup(<GalleryUiAdapterProvider>{null}</GalleryUiAdapterProvider>);
  return adapter;
};
const activeProject = () => store.getSnapshot().activeProject;
const previewId = () =>
  Object.values(activeProject().widgetInstances).find((instance) => instance.typeId === 'preview')!.id;
const regionSummary = () => {
  const { center, right } = activeProject().widgetRegions;
  return {
    center: { active: center.activeInstanceId, ids: center.instanceIds },
    right: { active: right.activeInstanceId, ids: right.instanceIds },
  };
};
const values = (projectId: string) => getProjectWidgetValues(store.queries.getProject(projectId)!, 'gallery');

beforeEach(() => {
  vi.clearAllMocks();
  accountLifecycle.activate('gallery-adapter-test');
  store = createWorkbenchStore();
});
afterEach(() => accountLifecycle.invalidate());

describe('Gallery live-follow adapter', () => {
  it('passes the followed session through and reveals Preview only for a tile click', () => {
    const owner = renderAdapter();

    expect(owner.followedProgressSessionId).toBe('run:1');

    const { center } = activeProject().widgetRegions;
    const galleryView = center.instanceIds.find((id) => id !== previewId())!;
    store.commands.widgets.select({ region: 'center', widgetId: galleryView });

    owner.followProgressSession('run:2', { revealPreview: false });
    expect(livePreviewFollow).toHaveBeenCalledExactlyOnceWith('run:2');
    expect(activeProject().widgetRegions.center.activeInstanceId).toBe(galleryView);

    owner.followProgressSession('run:2', { revealPreview: true });
    expect(activeProject().widgetRegions.center.activeInstanceId).toBe(previewId());
  });

  it('reveals a rail-docked Preview in its rail and leaves the Gallery in the center', () => {
    const preview = previewId();
    const { center, right } = activeProject().widgetRegions;
    const galleryView = center.instanceIds.find((id) => id !== preview)!;
    const otherRailTab = right.instanceIds[0]!;
    store.commands.widgets.move({ fromRegion: 'center', instanceId: preview, toIndex: 0, toRegion: 'right' });
    store.commands.widgets.select({ region: 'right', widgetId: otherRailTab });
    const before = regionSummary();
    expect(before.center).toEqual({ active: galleryView, ids: [galleryView] });
    expect(before.right.active).toBe(otherRailTab);

    renderAdapter().followProgressSession('run:2', { revealPreview: true });

    expect(regionSummary()).toEqual({
      ...before,
      right: { ...before.right, active: preview },
    });
  });

  it('unshades a floating Preview without docking it', () => {
    const preview = previewId();
    const galleryView = activeProject().widgetRegions.center.instanceIds.find((id) => id !== preview)!;
    store.commands.widgets.select({ region: 'center', widgetId: galleryView });
    store.commands.widgets.float(preview, 'center');
    store.commands.widgets.setFloatingMode(preview, 'shaded');
    const before = regionSummary();

    renderAdapter().followProgressSession('run:2', { revealPreview: true });

    expect(activeProject().floatingWidgets?.[preview]?.mode).toBe('windowed');
    expect(regionSummary()).toEqual(before);
  });

  it('opens Preview in the center when the layout has none', () => {
    const preview = previewId();
    for (const region of ['center', 'right', 'left', 'bottom'] as const) {
      if (activeProject().widgetRegions[region].instanceIds.includes(preview)) {
        store.commands.widgets.toggle({ region, widgetId: preview });
      }
    }
    expect(Object.values(activeProject().widgetRegions).some((region) => region.instanceIds.includes(preview))).toBe(
      false
    );

    renderAdapter().followProgressSession('run:2', { revealPreview: true });

    expect(activeProject().widgetRegions.center.activeInstanceId).toBe(previewId());
  });

  it('shows live frames in the Gallery only while Auto-switch is In progress', () => {
    const modes = (['progress', 'latest', 'off'] as const).map((mode) => {
      store.commands.account.updateProjectPreferences({ viewerAutoSwitch: mode });
      return renderAdapter().liveFollowEnabled;
    });

    expect(modes).toEqual([true, false, false]);
  });

  it('ignores a follow retained across an account change', () => {
    const owner = renderAdapter();

    accountLifecycle.activate('gallery-adapter-test');
    owner.followProgressSession('run:2', { revealPreview: true });
    expect(livePreviewFollow).not.toHaveBeenCalled();
  });
});

describe('Gallery settings adapter ownership', () => {
  it('updates the captured active project through the Gallery command', () => {
    const owner = renderAdapter();
    owner.gallery.updateSettings({ imageDensityPercent: 24, paginationMode: 'paginated' });

    expect(values(owner.projectId)).toMatchObject({ imageDensityPercent: 24, paginationMode: 'paginated' });
  });

  it('rejects a retained callback after switching projects and accepts the new project binding', () => {
    const first = renderAdapter();
    const firstValues = values(first.projectId);
    const secondProject = store.commands.projects.create();
    const secondValues = values(secondProject.id);

    first.gallery.updateSettings({ imageDensityPercent: 24 });
    expect(values(first.projectId)).toEqual(firstValues);
    expect(values(secondProject.id)).toEqual(secondValues);

    const second = renderAdapter();
    second.gallery.updateSettings({ imageDensityPercent: 12 });
    expect(values(secondProject.id).imageDensityPercent).toBe(12);
    expect(values(first.projectId)).toEqual(firstValues);
  });

  it('rejects retained callbacks after the account lifetime changes even with the same project id', () => {
    const previous = renderAdapter();
    const before = values(previous.projectId);
    accountLifecycle.activate('gallery-adapter-test');

    previous.gallery.updateSettings({ showImageDimensions: true });
    expect(values(previous.projectId)).toEqual(before);

    renderAdapter().gallery.updateSettings({ showImageDimensions: true });
    expect(values(previous.projectId).showImageDimensions).toBe(true);
  });
});
