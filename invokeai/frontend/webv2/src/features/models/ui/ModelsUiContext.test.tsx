import { accountLifecycle } from '@platform/state/accountLifecycle';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelsUiProvider, useOpenAddModelsSearch, useOpenModelInManager } from './ModelsUiContext';
import { getModelsUiSnapshotForTests } from './uiStore';

const { navigate, router } = vi.hoisted(() => ({
  navigate: vi.fn(),
  router: { state: { location: { href: '/app?project=project-a' } } },
}));

vi.mock('@tanstack/react-router', () => ({ useNavigate: () => navigate, useRouter: () => router }));

let activeProjectId = 'project-a';
const adapter = {
  canManageModels: true,
  enableModelDescriptions: true,
  isProjectActive: (projectId: string) => activeProjectId === projectId,
  managerProjectId: 'project-a',
};

const captureActions = () => {
  const observe =
    vi.fn<
      (actions: {
        openDetails: ReturnType<typeof useOpenModelInManager>;
        openSearch: ReturnType<typeof useOpenAddModelsSearch>;
      }) => void
    >();
  const Consumer = () => {
    const openDetails = useOpenModelInManager();
    const openSearch = useOpenAddModelsSearch();
    observe({ openDetails, openSearch });
    return null;
  };

  renderToStaticMarkup(
    <ModelsUiProvider adapter={adapter}>
      <Consumer />
    </ModelsUiProvider>
  );
  const actions = observe.mock.calls[0]![0];
  return { openDetails: actions.openDetails!, openSearch: actions.openSearch! };
};

beforeEach(() => {
  accountLifecycle.activate('account-a');
  activeProjectId = 'project-a';
  navigate.mockClear();
  router.state.location = { href: '/app?project=project-a' };
});
afterEach(() => accountLifecycle.invalidate());

describe('deferred Model Manager navigation', () => {
  it.each(['details', 'search'] as const)('opens %s for the initiating project', async (action) => {
    const { openDetails, openSearch } = captureActions();
    if (action === 'details') {
      openDetails('model-a');
    } else {
      openSearch('encoder-a');
    }
    await vi.dynamicImportSettled();

    expect(navigate).toHaveBeenCalledWith({ search: { project: 'project-a' }, to: '/models' });
    expect(getModelsUiSnapshotForTests()).toMatchObject(
      action === 'details'
        ? { activeModelKey: 'model-a', activeTab: 'details' }
        : { addModelsSeed: 'encoder-a', activeTab: 'add' }
    );
  });

  it.each(['account', 'location', 'project'] as const)('drops pending actions after the %s changes', async (change) => {
    const { openDetails, openSearch } = captureActions();
    openDetails('model-a');
    openSearch('encoder-a');

    if (change === 'account') {
      accountLifecycle.activate('account-b');
    } else if (change === 'project') {
      activeProjectId = 'project-b';
    } else {
      router.state.location = { href: '/app?project=project-b' };
    }
    await vi.dynamicImportSettled();

    expect(navigate).not.toHaveBeenCalled();
    expect(getModelsUiSnapshotForTests()).toMatchObject({ activeModelKey: null, addModelsSeed: null });
  });
});
