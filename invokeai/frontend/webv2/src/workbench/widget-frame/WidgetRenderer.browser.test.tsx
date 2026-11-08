import type {
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetImplementation,
  WidgetInstanceContract,
} from '@workbench/widgetContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as workbenchContext from '@workbench/WorkbenchContext';

import { Box, ChakraProvider, Stack, Text } from '@chakra-ui/react';
import { getLogSnapshot } from '@platform/logging/logger';
import { StatusWidgetChip } from '@platform/ui';
import { toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { FocusRegionProvider, useFocusRegionProps } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { WorkbenchReloadError } from '@workbench/projects/reloadWorkbench';
import { createWidgetRecovery } from '@workbench/widget-frame/widgetRecoveryStore';
import { createWidgetImplementationResource } from '@workbench/widgetImplementationResource';
import i18next from 'i18next';
import { CircleIcon } from 'lucide-react';
import { act, Activity, useState, type ReactNode, type SVGProps } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

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
  reload: vi.fn(() => Promise.resolve()),
}));

const recoveryOwner = vi.hoisted(() => ({
  widgetRecovery: new Map(),
  getSnapshot: () => ({ activeProject: { ...workbenchMocks.project, widgetInstances: { 'test-instance': {} } } }),
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  useWorkbenchInternalStore: () => recoveryOwner,
  useWorkbenchReload: () => workbenchMocks.reload,
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

const recoveryCopy = (await fetch('/locales/en.json').then((r) => r.json())).widgets.failure;
const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { cancel: 'Cancel' },
        widgets: {
          actionsLabel: '{{label}} actions',
          failure: {
            ...recoveryCopy,
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

beforeEach(() => {
  workbenchMocks.project.id = 'test-project';
  workbenchMocks.reload.mockReset().mockResolvedValue(undefined);
  recoveryOwner.widgetRecovery.set('test-instance', createWidgetRecovery());
});

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

  it.each(['left', 'center'] as const)(
    'refocuses the %s error host when a retried import rejects again',
    async (region) => {
      const { widget } = createDeferredWidget();
      const load = vi.fn().mockRejectedValue(new Error('Native-style repeated rejection'));
      widget.implementation = createWidgetImplementationResource('test', load);
      const view = (
        <WidgetRenderer instance={createInstance()} presentation="expanded" region={region} widget={widget} />
      );
      await render(region === 'center' ? <CenterFrame>{view}</CenterFrame> : view);
      await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      const retry = [...host!.querySelectorAll<HTMLButtonElement>('button')].find(
        (button) => button.textContent === 'Retry'
      )!;
      retry.focus();
      await userEvent.keyboard('{Enter}');
      await expect.poll(() => load.mock.calls.length).toBe(2);
      await expect.poll(() => host!.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      await expect
        .poll(() => document.activeElement)
        .toBe(
          [...host!.querySelectorAll<HTMLElement>(`[data-focus-region="${region}"]`)].find(
            (frame) => frame.getBoundingClientRect().height > 0
          )
        );
      const frame = document.activeElement!;
      await userEvent.keyboard('{Tab}');
      expect(frame.contains(document.activeElement)).toBe(true);
      expect(host!.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      if (region === 'left') {
        expect(frame.getAttribute('aria-label')).toBe('Test widget failed');
      }
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
    await expect.poll(() => host?.querySelectorAll('[data-testid="widget-failure"]').length).toBe(1);
    await userEvent.click(
      [...host!.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === 'Retry')!
    );
    await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).toBeNull();
    expect(load).toHaveBeenCalledTimes(2);
    expect(host?.querySelectorAll('button').length).toBeGreaterThanOrEqual(2);
    delete (workbenchMocks.project.widgetInstances as Record<string, unknown>)['test-instance'];
  });

  it('keeps a healthy body and its local draft through deterministic chrome failure and a failed-again retry', async () => {
    const Body = () => {
      const [draft, setDraft] = useState('');
      return <input aria-label="Local draft" value={draft} onChange={(event) => setDraft(event.target.value)} />;
    };
    const Actions = () => {
      throw new Error('deterministic chrome failure');
    };
    const { widget, resolve } = createDeferredWidget();
    Object.assign(workbenchMocks.project.widgetInstances, { 'test-instance': createInstance() });
    await render(
      <>
        <WidgetChromeSlotById instanceId="test-instance" widget={widget} slot="actions" />
        <WidgetRenderer instance={createInstance()} region="center" widget={widget} />
      </>
    );
    await resolve({ headerActions: Actions, view: Body });
    const input = host!.querySelector<HTMLInputElement>('input')!;
    await userEvent.fill(input, 'Held draft');
    await userEvent.click(
      host!.querySelector<HTMLButtonElement>('button[aria-label="Test widget failed — select to retry"]')!
    );
    await expect
      .poll(() => host!.querySelector('button[aria-label="Test widget failed — select to retry"]'))
      .not.toBeNull();
    expect(host!.querySelector('input')).toBe(input);
    expect(input.value).toBe('Held draft');
    expect(host!.querySelectorAll('[data-testid="widget-failure"]')).toHaveLength(1);
    delete (workbenchMocks.project.widgetInstances as Record<string, unknown>)['test-instance'];
  });

  it.each(['popover', 'dialog'] as const)(
    'keeps a named %s focus owner through a failed-again import retry',
    async (region) => {
      const { widget } = createDeferredWidget();
      const load = vi.fn().mockRejectedValue(new Error('still unavailable'));
      widget.implementation = createWidgetImplementationResource('test', load);
      await render(
        <WidgetRenderer instance={createInstance()} presentation="expanded" region={region} widget={widget} />
      );
      await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      const group = host!.querySelector<HTMLElement>('[data-widget-recovery-host]')!;
      expect(group.getAttribute('aria-label')).toBe('Test widget');
      const retry = [...host!.querySelectorAll<HTMLButtonElement>('button')].find(
        (button) => button.textContent === 'Retry'
      )!;
      retry.focus();
      await userEvent.keyboard('{Enter}');
      await expect.poll(() => load.mock.calls.length).toBe(2);
      await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      expect(host!.querySelector('[data-widget-recovery-host]')).toBe(group);
      expect(document.activeElement).toBe(group);
      await userEvent.keyboard('{Tab}');
      expect(group.contains(document.activeElement)).toBe(true);
      if (document.activeElement?.textContent !== 'Retry') {
        await userEvent.keyboard('{Tab}');
      }
      expect(document.activeElement?.textContent).toBe('Retry');
    }
  );

  it.each(['unmount', 'hide-and-reveal'] as const)(
    'does not reload when a pending retry rejects after %s',
    async (transition) => {
      let reject!: (error: Error) => void;
      const { widget } = createDeferredWidget();
      const load = vi
        .fn()
        .mockRejectedValueOnce(new Error('first import failure'))
        .mockImplementationOnce(
          () =>
            new Promise<WidgetImplementation>((_resolve, rejectLoad) => {
              reject = rejectLoad;
            })
        );
      widget.implementation = createWidgetImplementationResource('test', load);
      const Retained = ({ children }: { children: ReactNode }) => {
        const [visible, setVisible] = useState(true);
        return (
          <>
            <button type="button" onClick={() => setVisible(false)}>
              Hide recovery
            </button>
            <button type="button" onClick={() => setVisible(true)}>
              Show recovery
            </button>
            <Activity mode={visible ? 'visible' : 'hidden'}>{children}</Activity>
          </>
        );
      };
      await render(
        <Retained>
          <WidgetRenderer instance={createInstance()} presentation="expanded" region="center" widget={widget} />
        </Retained>
      );
      await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      await userEvent.click(
        [...host!.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === 'Retry')!
      );
      await expect.poll(() => load.mock.calls.length).toBe(2);
      if (transition === 'unmount') {
        await act(() => root!.unmount());
        root = null;
      } else {
        await userEvent.click(
          [...host!.querySelectorAll<HTMLButtonElement>('button')].find(
            (button) => button.textContent === 'Hide recovery'
          )!
        );
        await userEvent.click(
          [...host!.querySelectorAll<HTMLButtonElement>('button')].find(
            (button) => button.textContent === 'Show recovery'
          )!
        );
      }
      await act(async () => {
        reject(new Error('late repeated rejection'));
        await Promise.resolve();
      });
      if (transition !== 'unmount') {
        await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      }
      expect(workbenchMocks.reload).not.toHaveBeenCalled();
      expect(load).toHaveBeenCalledTimes(2);
    }
  );

  it('keeps Retry in place and requires a separately confirmed editor reload', async () => {
    let finish!: () => void;
    workbenchMocks.reload.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        })
    );
    const { widget } = createDeferredWidget();
    const load = vi.fn().mockRejectedValue(new Error('deterministic import rejection'));
    widget.implementation = createWidgetImplementationResource('test', load);
    await render(
      <WidgetRenderer instance={createInstance()} presentation="expanded" region="center" widget={widget} />
    );
    const retry = page.getByRole('button', { name: 'Retry', exact: true });
    await retry.click();
    await expect.poll(() => load.mock.calls.length).toBe(2);
    await expect.element(page.getByRole('button', { name: 'Reload editor', exact: true })).toBeVisible();
    expect(workbenchMocks.reload).not.toHaveBeenCalled();
    await retry.click();
    await expect.poll(() => load.mock.calls.length).toBe(3);
    expect(workbenchMocks.reload).not.toHaveBeenCalled();
    const reloadButton = page.getByRole('button', { name: 'Reload editor', exact: true });
    await reloadButton.click();
    const dialog = page.getByRole('alertdialog', { name: 'Reload the editor?' });
    await expect.element(dialog).toBeVisible();
    expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('ALL open projects');
    expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain(
      'Canvas undo and per-workflow edit histories'
    );
    await expect.element(dialog.getByRole('button', { name: 'Cancel', exact: true })).toHaveFocus();
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
    expect(workbenchMocks.reload).not.toHaveBeenCalled();
    await expect.element(reloadButton).toHaveFocus();
    await reloadButton.click();
    const confirm = dialog.getByRole('button', { name: 'Reload editor', exact: true });
    await confirm.click();
    await expect.poll(() => workbenchMocks.reload.mock.calls.length).toBe(1);
    await expect.element(confirm).toBeDisabled();
    expect(load).toHaveBeenCalledTimes(3);
    await act(() => finish());
    await expect.element(dialog).not.toBeInTheDocument();
  });

  it.each(['changed', 'pixels'] as const)(
    'reports a %s refusal in localized copy and restores focus to the explicit action',
    async (reason) => {
      const toast = vi.spyOn(toaster, 'create').mockImplementation(() => 'reload-error');
      workbenchMocks.reload.mockRejectedValueOnce(
        reason === 'changed'
          ? new WorkbenchReloadError('changed')
          : new AggregateError([new Error('Pixel encoder diagnostic')], 'Unsaved Canvas pixels could not be persisted.')
      );
      const { widget } = createDeferredWidget();
      widget.implementation = createWidgetImplementationResource('test', () =>
        Promise.reject(new Error('unavailable'))
      );
      await render(
        <WidgetRenderer instance={createInstance()} presentation="expanded" region="center" widget={widget} />
      );
      await page.getByRole('button', { name: 'Retry', exact: true }).click();
      const reload = page.getByRole('button', { name: 'Reload editor', exact: true });
      await reload.click();
      const dialog = page.getByRole('alertdialog');
      await dialog.getByRole('button', { name: 'Reload editor', exact: true }).click();
      await expect.element(dialog).not.toBeInTheDocument();
      await expect.element(reload).toHaveFocus();
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'error',
          title: recoveryCopy.reloadFailed,
          description: recoveryCopy.reloadReasons[reason],
        })
      );
      expect(recoveryCopy.reloadReasons.changed).toContain('background updates');
      if (reason === 'pixels') {
        const entry = getLogSnapshot().entries.find((entry) => entry.name === 'widget.reload-refused');
        expect(entry?.context).toMatchObject({
          pixelFailures: [{ name: 'Error', message: 'Pixel encoder diagnostic' }],
        });
      }
      toast.mockRestore();
    }
  );

  it.each(['project', 'account', 'hide-and-reveal'] as const)(
    'fences a reload confirmation after %s changes',
    async (transition) => {
      const toast = vi.spyOn(toaster, 'create').mockImplementation(() => 'inactive-reload');
      const { widget } = createDeferredWidget();
      widget.implementation = createWidgetImplementationResource('test', () =>
        Promise.reject(new Error('unavailable'))
      );
      const content = (
        <WidgetRenderer instance={createInstance()} presentation="expanded" region="center" widget={widget} />
      );
      const Retained = ({ children }: { children: ReactNode }) => {
        const [visible, setVisible] = useState(true);
        return (
          <>
            <button type="button" onClick={() => setVisible(false)}>
              Hide confirmation owner
            </button>
            <button type="button" onClick={() => setVisible(true)}>
              Show confirmation owner
            </button>
            <Activity mode={visible ? 'visible' : 'hidden'}>{children}</Activity>
          </>
        );
      };
      await render(<Retained>{content}</Retained>);
      await page.getByRole('button', { name: 'Retry', exact: true }).click();
      await page.getByRole('button', { name: 'Reload editor', exact: true }).click();
      if (transition === 'hide-and-reveal') {
        await act(() =>
          [...host!.querySelectorAll<HTMLButtonElement>('button')]
            .find((button) => button.textContent === 'Hide confirmation owner')!
            .click()
        );
        await act(() =>
          [...host!.querySelectorAll<HTMLButtonElement>('button')]
            .find((button) => button.textContent === 'Show confirmation owner')!
            .click()
        );
      } else if (transition === 'project') {
        workbenchMocks.project.id = 'replacement-project';
      } else {
        const { accountLifecycle } = await import('@platform/state/accountLifecycle');
        accountLifecycle.activate('replacement', ':user:replacement');
      }
      await page.getByRole('alertdialog').getByRole('button', { name: 'Reload editor', exact: true }).click();
      expect(workbenchMocks.reload).not.toHaveBeenCalled();
      if (transition === 'hide-and-reveal') {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({ description: recoveryCopy.reloadReasons.inactive })
        );
      } else {
        expect(toast).not.toHaveBeenCalled();
      }
      toast.mockRestore();
      workbenchMocks.project.id = 'test-project';
    }
  );

  it('lets non-activation workbench chords bubble from compact Retry', async () => {
    const { widget, resolve } = createDeferredWidget();
    await render(<WidgetRenderer instance={createInstance()} presentation="compact" region="bottom" widget={widget} />);
    await resolve({ view: ThrowingView });
    const events: string[] = [];
    const listen = (event: KeyboardEvent) => {
      events.push(event.key);
    };
    window.addEventListener('keydown', listen);
    try {
      host!.querySelector<HTMLButtonElement>('button')!.focus();
      await userEvent.keyboard('{Alt>}{Shift>}{ArrowRight}{/Shift}{/Alt}');
      await userEvent.keyboard('{Alt>}{PageDown}{/Alt}');
      await userEvent.keyboard('{Control>}k{/Control}');
      expect(events).toEqual(expect.arrayContaining(['ArrowRight', 'PageDown', 'k']));
    } finally {
      window.removeEventListener('keydown', listen);
    }
  });

  it.each(['{Enter}', '[Space]'])(
    'recovers failed chrome with one %s retry while its healthy body survives',
    async (key) => {
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
      await expect.poll(() => host?.querySelectorAll('[data-testid="widget-failure"]').length).toBe(1);
      const retry = host!.querySelector<HTMLButtonElement>(
        'button[aria-label="Test widget failed — select to retry"]'
      )!;
      retry.focus();
      expect(document.activeElement).toBe(retry);
      broken = false;
      await userEvent.keyboard(key);
      await expect.poll(() => host?.querySelector('[data-testid="widget-failure"]')).toBeNull();
      expect(retryResource).toHaveBeenCalledTimes(1);
      expect(host?.textContent).toContain('Recovered action');
      expect(host?.textContent).toContain('Loaded control');
      delete (workbenchMocks.project.widgetInstances as Record<string, unknown>)['test-instance'];
    }
  );
});
