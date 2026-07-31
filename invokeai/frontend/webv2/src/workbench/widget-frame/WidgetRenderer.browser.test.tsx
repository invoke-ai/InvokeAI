/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type {
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetImplementation,
  WidgetInstanceContract,
} from '@workbench/widgetContracts';

import { ChakraProvider, Stack, Text } from '@chakra-ui/react';
import { StatusWidgetChip } from '@platform/ui';
import { system } from '@theme/system';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { createWidgetImplementationResource } from '@workbench/widgetImplementationResource';
import i18next from 'i18next';
import { CircleIcon } from 'lucide-react';
import { act, type ReactNode, type SVGProps } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

const workbenchMocks = vi.hoisted(() => ({
  project: {
    id: 'test-project',
    invocation: { sourceId: 'test' },
    projectGraph: { nodes: [] },
    widgetGraphs: {},
    widgetInstances: {},
    widgetRegions: {
      bottom: { activeInstanceId: 'test-instance', instanceIds: ['test-instance'], isCollapsed: false, sizePx: 240 },
      center: { activeInstanceId: 'test-instance', instanceIds: ['test-instance'], isCollapsed: false, sizePx: 240 },
      left: { activeInstanceId: 'test-instance', instanceIds: ['test-instance'], isCollapsed: false, sizePx: 240 },
      right: { activeInstanceId: 'test-instance', instanceIds: ['test-instance'], isCollapsed: false, sizePx: 240 },
    },
  },
  runtime: {},
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  shallowEqual: Object.is,
  useActiveProjectSelector: (selector: (project: typeof workbenchMocks.project) => unknown) =>
    selector(workbenchMocks.project),
  useWorkbenchCommands: () => ({ layout: { setRegionSize: () => {} } }),
}));

vi.mock('@workbench/WorkbenchWidgetRegistryContext', () => ({
  useWorkbenchWidgetRegistry: () => ({
    getWidgetById: () => undefined,
    getWidgetsForRegion: () => [],
  }),
}));

vi.mock('./createWidgetRuntime', () => ({
  useWidgetRuntime: () => workbenchMocks.runtime,
}));

import { WidgetRenderer } from './WidgetRenderer';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          actionsLabel: '{{label}} actions',
          failure: {
            compactLabel: '{{label}} failed — select to retry',
            copyError: 'Copy Error',
            retry: 'Retry',
            title: '{{label}} failed',
          },
          loadingLabel: 'Loading {{label}}',
          panelLabel: '{{region}} widget panel',
        },
      },
    },
  },
});

const TestIcon = (props: SVGProps<SVGSVGElement>) => <svg {...props} />;
const TestView = () => <div data-testid="loaded" />;
const TooltipView = () => (
  <Stack data-testid="loaded" gap="2">
    <Text data-widget-identity-label="" fontSize="xs" fontWeight="700">
      Test widget
    </Text>
    <Text color="fg.subtle" fontSize="2xs">
      Ready
    </Text>
  </Stack>
);
const CompactView = () => (
  <StatusWidgetChip icon={CircleIcon}>
    <span data-widget-identity-label="">Test widget</span>
  </StatusWidgetChip>
);
const CustomHeaderLabel = () => (
  <Text data-widget-identity-label="" fontSize="xs" fontWeight="700">
    Custom widget label
  </Text>
);

const ThrowingView = () => {
  throw new Error('widget exploded');
};

const createDeferredWidget = () => {
  let resolveLoad: ((implementation: WidgetImplementation) => void) | undefined;
  const loader = vi.fn(
    () =>
      new Promise<WidgetImplementation>((resolve) => {
        resolveLoad = resolve;
      })
  );
  const manifest: NormalizedWidgetManifest = {
    allowMultiple: false,
    allowedRegions: ['bottom', 'center', 'right'],
    apiVersion: 1,
    failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    icon: TestIcon,
    id: 'test',
    label: 'Test widget',
    load: loader,
    state: { createInitial: () => ({}), persistence: 'project', version: 1 },
    version: 1,
  };
  const widget: RegisteredWidget = {
    implementation: createWidgetImplementationResource('test', loader),
    manifest,
    status: 'enabled',
  };

  return {
    resolve: async (implementation: WidgetImplementation) => {
      await act(async () => {
        resolveLoad?.(implementation);
        await Promise.resolve();
      });
    },
    widget,
  };
};

const createInstance = (title?: string): WidgetInstanceContract => ({
  createdAt: '2026-07-19T00:00:00.000Z',
  id: 'test-instance',
  state: { id: 'test', label: 'Test widget', values: {}, version: 1 },
  title,
  typeId: 'test',
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (children: ReactNode) => {
  host = document.createElement('div');
  host.style.cssText = 'height:320px;width:480px;';
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <FocusRegionProvider>{children}</FocusRegionProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });
};

const getIdentityGeometry = ({ loadedIcon = false } = {}) => {
  const icon = host?.querySelector<HTMLElement>(loadedIcon ? 'svg' : '[data-widget-identity-slot]');
  const label = host?.querySelector<HTMLElement>('[data-widget-identity-label]');

  expect(icon).not.toBeNull();
  expect(label).not.toBeNull();
  if (!icon || !label) {
    throw new Error('Expected widget identity icon and label.');
  }

  const iconRect = icon.getBoundingClientRect();
  const labelRect = label.getBoundingClientRect();
  expect(iconRect.width).toBe(12);
  expect(labelRect.left).toBeGreaterThan(iconRect.right);

  return {
    iconLeft: iconRect.left,
    iconWidth: iconRect.width,
    labelLeft: labelRect.left,
  };
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

describe('WidgetRenderer loading identity transitions', () => {
  // The center region has no header row of its own — `CenterArea` floats that
  // chrome above the work surface — so the identity slot lives in panel frames.
  it('swaps spinner for icon without moving a renamed standard header', async () => {
    const { resolve, widget } = createDeferredWidget();
    await render(<WidgetRenderer instance={createInstance('Renamed widget')} region="right" widget={widget} />);

    const loadingGeometry = getIdentityGeometry();
    expect(host?.textContent).toContain('Renamed widget');

    await resolve({ view: TestView });
    await expect.poll(() => host?.querySelector('[data-testid="loaded"]')).not.toBeNull();

    expect(getIdentityGeometry()).toEqual(loadingGeometry);
    expect(host?.textContent).toContain('Renamed widget');
    expect(host?.textContent).not.toContain('Test widget');
    expect(host?.querySelector('.chakra-spinner')).toBeNull();
  });

  it('keeps the identity slot aligned when a custom header label loads', async () => {
    const { resolve, widget } = createDeferredWidget();
    await render(<WidgetRenderer instance={createInstance()} region="right" widget={widget} />);

    const loadingGeometry = getIdentityGeometry();
    await resolve({ headerLabel: CustomHeaderLabel, view: TestView });
    await expect.poll(() => host?.querySelector('[data-testid="loaded"]')).not.toBeNull();

    expect(getIdentityGeometry()).toEqual(loadingGeometry);
    expect(host?.textContent).toContain('Custom widget label');
  });

  it('preserves spinner and icon geometry for compact widgets', async () => {
    const { resolve, widget } = createDeferredWidget();
    await render(<WidgetRenderer instance={createInstance()} presentation="compact" region="bottom" widget={widget} />);

    const loadingGeometry = getIdentityGeometry();
    await resolve({ view: CompactView });
    await expect.poll(() => host?.querySelector('[data-widget-identity-label]')).not.toBeNull();

    expect(getIdentityGeometry({ loadedIcon: true })).toEqual(loadingGeometry);
    expect(host?.querySelector('.chakra-spinner')).toBeNull();
  });

  it('preserves the identity slot when tooltip details load', async () => {
    const { resolve, widget } = createDeferredWidget();
    await render(<WidgetRenderer instance={createInstance()} presentation="tooltip" region="bottom" widget={widget} />);

    const loadingGeometry = getIdentityGeometry();
    await resolve({ view: TooltipView });
    await expect.poll(() => host?.querySelector('[data-testid="loaded"]')).not.toBeNull();

    expect(getIdentityGeometry()).toEqual(loadingGeometry);
    expect(host?.querySelector('.chakra-spinner')).toBeNull();
  });
});

describe('WidgetRenderer failure containment', () => {
  it('keeps a crashed compact widget inside the status bar strip', async () => {
    const { resolve, widget } = createDeferredWidget();
    // The status bar is a 24px-tall strip; a crashed widget must not outgrow it.
    host = document.createElement('div');
    host.style.cssText = 'align-items:center;display:flex;height:24px;width:480px;';
    document.body.append(host);
    root = createRoot(host);

    await act(async () => {
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <WidgetRenderer instance={createInstance()} presentation="compact" region="bottom" widget={widget} />
          </ChakraProvider>
        </I18nextProvider>
      );
      await Promise.resolve();
    });

    await resolve({ view: ThrowingView });
    await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();

    const failure = host?.querySelector<HTMLElement>('[role="status"]');
    expect(failure).not.toBeNull();
    expect(failure?.getBoundingClientRect().height).toBeLessThanOrEqual(24);
    // The label still identifies which widget died.
    expect(host?.textContent).toContain('Test widget');
  });

  it('keeps a crashed panel widget framed and resizable', async () => {
    const { resolve, widget } = createDeferredWidget();
    await render(<WidgetRenderer instance={createInstance()} region="right" widget={widget} />);

    await resolve({ view: ThrowingView });
    await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();

    const panel = host?.querySelector<HTMLElement>('aside');
    expect(panel).not.toBeNull();
    // Panel width survives the crash, so the region does not collapse or overflow.
    expect(panel?.getBoundingClientRect().width).toBe(240);
    // And the drag handle is still there to resize with.
    expect(panel?.querySelector('[role="separator"]')).not.toBeNull();
  });
});
