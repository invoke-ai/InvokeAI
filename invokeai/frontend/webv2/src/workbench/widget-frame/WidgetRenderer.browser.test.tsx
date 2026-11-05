import type {
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetImplementation,
  WidgetInstanceContract,
} from '@workbench/widgetContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as workbenchContext from '@workbench/WorkbenchContext';

import { Box, ChakraProvider, Stack, Text } from '@chakra-ui/react';
import { StatusWidgetChip } from '@platform/ui';
import { system } from '@theme/system';
import { FocusRegionProvider, useFocusRegionProps } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { createWidgetImplementationResource } from '@workbench/widgetImplementationResource';
import i18next from 'i18next';
import { CircleIcon } from 'lucide-react';
import { act, type ReactNode, type SVGProps } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const workbenchMocks = vi.hoisted(() => ({
  project: {
    id: 'test-project',
    invocation: { sourceId: 'test' },
    projectGraph: { nodes: [] },
    widgetGraphs: {},
    widgetInstances: {},
    widgetRegions: {
      bottom: { activeInstanceId: 'test-instance', instanceIds: ['test-instance'], isCollapsed: false, sizePx: 384 },
      center: { activeInstanceId: 'test-instance', instanceIds: ['test-instance'], isCollapsed: false, sizePx: 384 },
      left: { activeInstanceId: 'test-instance', instanceIds: ['test-instance'], isCollapsed: false, sizePx: 384 },
      right: { activeInstanceId: 'test-instance', instanceIds: ['test-instance'], isCollapsed: false, sizePx: 384 },
    },
  },
  runtime: {},
}));

const recoveryOwner = vi.hoisted(() => ({
  widgetRecovery: new Map(),
  getSnapshot: () => ({ activeProject: { ...workbenchMocks.project, widgetInstances: { 'test-instance': {} } } }),
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  useWorkbenchInternalStore: () => recoveryOwner,
  shallowEqual: Object.is,
  useActiveProjectId: () => workbenchMocks.project.id,
  useWorkbenchQueries: () => ({
    getProject: (projectId: string) => (projectId === workbenchMocks.project.id ? workbenchMocks.project : null),
    isActiveProject: (projectId: string) => projectId === workbenchMocks.project.id,
  }),
  useActiveProjectSelector: (selector: (project: typeof workbenchMocks.project) => unknown) =>
    selector(workbenchMocks.project),
  useWorkbenchCommands: () => ({ layout: { setRegionCollapsed: () => {}, setRegionSize: () => {} } }),
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

import { WidgetChromeSlotById, WidgetRenderer } from './WidgetRenderer';

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
const FocusableView = () => <button type="button">Loaded control</button>;
const CenterFrame = ({ children }: { children: ReactNode }) => <Box {...useFocusRegionProps('center')}>{children}</Box>;
const TooltipView = () => (
  <Stack data-testid="loaded" gap="2">
    <Text data-widget-identity-label="" fontSize="md" fontWeight="700">
      Test widget
    </Text>
    <Text color="fg.subtle" fontSize="xs">
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
  <Text data-widget-identity-label="" fontSize="md" fontWeight="700">
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
    allowedRegions: ['bottom', 'center', 'left', 'right'],
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

const focusController = createTestFocusController();

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
          <FocusRegionProvider controller={focusController}>{children}</FocusRegionProvider>
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
  recoveryOwner.widgetRecovery.clear();
  // Focus is per test: the next one starts with no region outlined and no move pending.
  focusController.clear();
  await act(async () => {
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = null;
  root = null;
});

describe('WidgetRenderer loading identity transitions', () => {
  it.each(['left', 'center'] as const)(
    'keeps %s navigation focus when a lazy view replaces its fallback',
    async (region) => {
      const { resolve, widget } = createDeferredWidget();
      widget.manifest.chrome = { header: 'hidden' };
      const view = <WidgetRenderer instance={createInstance()} region={region} widget={widget} />;
      await render(
        <>
          <button type="button">Opener</button>
          {region === 'center' ? <CenterFrame>{view}</CenterFrame> : view}
        </>
      );
      const opener = host!.querySelector('button')!;
      opener.focus();
      focusController.focusRegion(region, undefined, 'test-instance');
      await act(async () => {
        await new Promise(requestAnimationFrame);
        await new Promise(requestAnimationFrame);
      });
      // A replaceable side fallback must not take focus; center has a persistent owning frame.
      if (region === 'left') {
        expect(document.activeElement).toBe(opener);
      } else {
        expect(document.activeElement).toBe(host!.querySelector('[data-focus-region="center"]'));
      }
      await resolve({ view: FocusableView });
      await expect.poll(() => host?.textContent).toContain('Loaded control');
      await expect.poll(() => document.activeElement).toBe(host!.querySelector(`[data-focus-region="${region}"]`));
      await act(() => userEvent.keyboard('{Tab}'));
      expect(document.activeElement?.textContent).toBe('Loaded control');
    }
  );

  // Center and right docks own separate chrome; identity slots belong in left/bottom panel frames.
  it('swaps spinner for icon without moving a renamed standard header', async () => {
    const { resolve, widget } = createDeferredWidget();
    await render(<WidgetRenderer instance={createInstance('Renamed widget')} region="left" widget={widget} />);

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
    await render(<WidgetRenderer instance={createInstance()} region="left" widget={widget} />);

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

    const failure = host?.querySelector<HTMLElement>('button');
    expect(failure).not.toBeNull();
    expect(failure?.getBoundingClientRect().height).toBeLessThanOrEqual(24);
    // The label still identifies which widget died.
    expect(host?.textContent).toContain('Test widget');
  });

  it('keeps a crashed panel widget framed and resizable', async () => {
    const { resolve, widget } = createDeferredWidget();
    await render(<WidgetRenderer instance={createInstance()} region="left" widget={widget} />);

    await resolve({ view: ThrowingView });
    await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();

    const frame = host?.querySelector<HTMLElement>('aside')?.parentElement;
    expect(frame).toBeDefined();
    // Panel width survives the crash, so the region does not collapse or overflow.
    expect(frame?.getBoundingClientRect().width).toBe(384);
    // And the drag handle is still there to resize with.
    expect(frame?.querySelector('[role="separator"]')).not.toBeNull();
  });
});

describe('shared widget recovery', () => {
  it.each(['left', 'center', 'floating'] as const)(
    'restores keyboard focus to the recovered %s host after expanded retry',
    async (region) => {
      const { widget, resolve } = createDeferredWidget();
      widget.implementation = createWidgetImplementationResource(
        'test',
        vi.fn().mockRejectedValueOnce(new Error('chunk unavailable')).mockImplementation(widget.manifest.load)
      );
      const retryResource = vi.spyOn(widget.implementation, 'retry');
      const view = <WidgetRenderer instance={createInstance()} region={region} widget={widget} />;
      await render(
        region === 'center' ? (
          <CenterFrame>{view}</CenterFrame>
        ) : region === 'floating' ? (
          <div data-floating-window="test-instance" tabIndex={-1}>
            {view}
          </div>
        ) : (
          view
        )
      );
      await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      const retry = [...host!.querySelectorAll<HTMLButtonElement>('button')].find(
        (button) => button.textContent === 'Retry'
      )!;
      retry.focus();
      await userEvent.keyboard('{Enter}');
      await expect.poll(() => retryResource.mock.calls.length).toBe(1);
      // Imports may take longer than the focus controller's bounded DOM-frame wait.
      await new Promise((resolveDelay) => {
        setTimeout(resolveDelay, 1200);
      });
      await resolve({ view: FocusableView });
      await expect.poll(() => host?.textContent).toContain('Loaded control');
      const selector = region === 'floating' ? '[data-floating-window]' : `[data-focus-region="${region}"]`;
      await expect.poll(() => document.activeElement).toBe(host!.querySelector(selector));
      await userEvent.keyboard('{Tab}');
      expect(document.activeElement?.textContent).toBe('Loaded control');
      expect(retryResource).toHaveBeenCalledTimes(1);
    }
  );

  it('refuses a stale retry after its project has left the owner', async () => {
    const { widget, resolve } = createDeferredWidget();
    const retryResource = vi.spyOn(widget.implementation, 'retry');
    await render(<WidgetRenderer instance={createInstance()} presentation="compact" region="bottom" widget={widget} />);
    await resolve({ view: ThrowingView });
    await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
    const previousProject = workbenchMocks.project.id;
    try {
      workbenchMocks.project.id = 'another-project';
      await userEvent.click(host!.querySelector<HTMLButtonElement>('button')!);
      expect(retryResource).not.toHaveBeenCalled();
      expect(host?.querySelector('[data-testid="loaded"]')).toBeNull();
    } finally {
      workbenchMocks.project.id = previousProject;
    }
  });

  it('retries a rejected implementation once for all slots', async () => {
    const { widget } = createDeferredWidget();
    const load = vi
      .fn()
      .mockRejectedValueOnce(new Error('chunk unavailable'))
      .mockResolvedValue({ headerActions: FocusableView, view: FocusableView });
    widget.implementation = createWidgetImplementationResource('test', load);
    Object.assign(workbenchMocks.project.widgetInstances, { 'test-instance': createInstance() });
    await render(
      <>
        <WidgetChromeSlotById instanceId="test-instance" widget={widget} slot="actions" />
        <WidgetRenderer instance={createInstance()} region="center" widget={widget} />
      </>
    );
    await expect.poll(() => host?.querySelectorAll('[data-testid="widget-failure"]').length).toBe(2);
    await userEvent.click(host!.querySelector<HTMLButtonElement>('button[aria-label="Retry: Test widget"]')!);
    await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).toBeNull();
    expect(load).toHaveBeenCalledTimes(2);
    expect(host?.querySelectorAll('button').length).toBeGreaterThanOrEqual(2);
    delete (workbenchMocks.project.widgetInstances as Record<string, unknown>)['test-instance'];
  });

  it.each(['{Enter}', '[Space]'])('resets failed chrome and body together with one %s retry', async (key) => {
    let broken = true;
    const HeaderActions = () => {
      if (broken) {
        throw new Error('chrome exploded');
      }
      return <button type="button">Recovered action</button>;
    };
    const { widget, resolve } = createDeferredWidget();
    const retryResource = vi.spyOn(widget.implementation, 'retry');
    Object.assign(workbenchMocks.project.widgetInstances, { 'test-instance': createInstance() });
    await render(
      <>
        <WidgetChromeSlotById instanceId="test-instance" widget={widget} slot="actions" />
        <WidgetRenderer instance={createInstance()} region="center" widget={widget} />
      </>
    );
    await resolve({ headerActions: HeaderActions, view: FocusableView });
    await expect.poll(() => host?.querySelectorAll('[data-testid="widget-failure"]').length).toBe(2);
    const retry = host!.querySelector<HTMLButtonElement>('button[aria-label="Retry: Test widget"]')!;
    retry.focus();
    expect(document.activeElement).toBe(retry);
    broken = false;
    await userEvent.keyboard(key);
    await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).toBeNull();
    expect(retryResource).toHaveBeenCalledTimes(1);
    expect(host?.textContent).toContain('Recovered action');
    expect(host?.textContent).toContain('Loaded control');
    delete (workbenchMocks.project.widgetInstances as Record<string, unknown>)['test-instance'];
  });
});
