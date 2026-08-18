/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { Box, ChakraProvider, HStack, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

const centerAreaMocks = vi.hoisted(() => {
  const icon = () => null;
  const activeItem = {
    icon,
    id: 'preview-instance',
    instance: { id: 'preview-instance' },
    label: 'Preview',
    status: 'enabled',
    typeId: 'preview',
    widget: {
      manifest: {
        centerPlacement: 'view',
        chrome: { header: 'visible' },
        icon,
      },
      status: 'enabled',
    },
  };

  return {
    activeItem,
    project: {
      id: 'test-project',
      invocation: { sourceId: 'generate' },
      queue: { items: [] },
      widgetInstances: {},
      widgetRegions: {
        center: { activeInstanceId: activeItem.id, instanceIds: [activeItem.id], isCollapsed: false, sizePx: 0 },
      },
    },
  };
});

vi.mock('@features/models', () => ({ useModelLoads: () => [] }));
vi.mock('@features/queue/contracts', () => ({
  getProjectQueueIndicatorState: () => ({ hasOpenQueueWork: false, progressState: null, runningQueueItemId: null }),
}));
vi.mock('@features/queue/react', () => ({ useQueueItemProgress: () => null }));
vi.mock('@workbench/focusRegions', () => ({ useFocusRegionProps: () => ({}) }));
vi.mock('@workbench/widgetRegionViewModel', () => ({
  createWidgetRegionViewModelFromState: () => ({ placedItems: [centerAreaMocks.activeItem] }),
  getWidgetRegionItems: () => [],
  isRequiredCenterView: () => true,
}));
vi.mock('@workbench/widgetRegistry', () => ({ getWidgetById: () => undefined, getWidgetsForRegion: () => [] }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => centerAreaMocks.project.id,
  useActiveProjectSelector: (selector: (project: typeof centerAreaMocks.project) => unknown) =>
    selector(centerAreaMocks.project),
  useWorkbenchCommands: () => ({ widgets: {} }),
  useWorkbenchSelector: (selector: (snapshot: { backendConnection: { status: string } }) => unknown) =>
    selector({ backendConnection: { status: 'connected' } }),
}));
vi.mock('@workbench/widget-frame', () => ({
  WidgetChromeSlotById: ({ slot }: { slot: 'actions' | 'label' }) => {
    if (slot === 'actions') {
      return <Box data-testid="trailing-actions" flexShrink={0} w="189px" />;
    }

    return (
      <HStack flex="1" minW="0" w="300px">
        <Text flexShrink={0}>Panther and the Flask</Text>
        <Text truncate>a423880b-184a-4910-977e-fe7d7fa9f732.png</Text>
      </HStack>
    );
  },
  WidgetRendererById: () => <Box h="full" w="full" />,
  WidgetSourceLockBadge: () => null,
  useWidgetIntentPreloadProps: () => ({}),
}));

import { CenterArea } from './CenterArea';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          centerRegion: 'Center view',
          centerViewLabel: 'Center view: {{label}}',
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (children: ReactNode) => {
  host = document.createElement('div');
  host.style.cssText = 'display:flex;height:320px;width:571px;';
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>{children}</ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });
};

afterEach(async () => {
  await act(async () => {
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = null;
  root = null;
});

describe('CenterArea chrome layout', () => {
  it('keeps non-shrinking trailing actions inside the center region', async () => {
    await render(<CenterArea />);

    const centerRegion = host?.querySelector<HTMLElement>('section');
    const trailingActions = host?.querySelector<HTMLElement>('[data-testid="trailing-actions"]');

    expect(centerRegion).not.toBeNull();
    expect(trailingActions).not.toBeNull();
    if (!centerRegion || !trailingActions) {
      throw new Error('Expected center chrome geometry.');
    }

    expect(trailingActions.getBoundingClientRect().right).toBeLessThanOrEqual(
      centerRegion.getBoundingClientRect().right
    );
  });
});
