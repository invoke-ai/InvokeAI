/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { NormalizedWidgetManifest, RegisteredWidget, WidgetInstanceContract } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createWidgetImplementationResource } from '@workbench/widgetImplementationResource';
import i18next from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { WidgetEnableMenu } from './WidgetEnableMenu';
import { WidgetLoadingFallback } from './WidgetLoadingFallback';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          loadingLabel: 'Loading {{label}}',
        },
      },
    },
  },
});

const TestIcon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />;
const TestView = () => null;

const createTestWidget = ({ hiddenHeader = false, loader = vi.fn(() => Promise.resolve({ view: TestView })) } = {}) => {
  const manifest: NormalizedWidgetManifest = {
    allowMultiple: false,
    allowedRegions: ['center'],
    apiVersion: 1,
    chrome: hiddenHeader ? { header: 'hidden' } : undefined,
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

  return { loader, widget };
};

const instance: WidgetInstanceContract = {
  createdAt: '2026-07-19T00:00:00.000Z',
  id: 'test-instance',
  state: { id: 'test', label: 'Test widget', values: {}, version: 1 },
  typeId: 'test',
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await Promise.resolve();
  });

const render = async (children: ReactNode) => {
  host = document.createElement('div');
  host.style.cssText = 'height:320px;width:480px;';
  document.body.append(host);
  root = createRoot(host);

  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>{children}</ChakraProvider>
      </I18nextProvider>
    );
  });
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  document.querySelectorAll('[data-scope="menu"]').forEach((element) => element.remove());
  host = null;
  root = null;
});

describe('WidgetLoadingFallback', () => {
  it('keeps center widget chrome and geometry stable while the implementation loads', async () => {
    const { widget } = createTestWidget();
    await render(<WidgetLoadingFallback instance={instance} region="center" widget={widget} />);

    const frame = host?.querySelector<HTMLElement>('[data-hotkey-widget-instance-id="test-instance"]');
    const busySurface = host?.querySelector<HTMLElement>('[aria-busy="true"]');

    expect(frame?.getBoundingClientRect().width).toBe(480);
    expect(frame?.getBoundingClientRect().height).toBe(320);
    expect(frame?.textContent).toBe('Test widget');
    expect(busySurface?.getAttribute('aria-label')).toBe('Loading Test widget');
    expect(busySurface?.getAttribute('aria-live')).toBe('polite');
    expect(busySurface?.getAttribute('role')).toBe('status');
    expect(host?.querySelector('.chakra-spinner')).not.toBeNull();
    expect(host?.textContent).not.toContain('Loading widget');
  });

  it('honors hidden chrome and uses the accessible loading label without visible loading copy', async () => {
    const { widget } = createTestWidget({ hiddenHeader: true });
    await render(<WidgetLoadingFallback instance={instance} region="center" widget={widget} />);

    expect(host?.textContent).toBe('');
    expect(host?.querySelector('[aria-busy="true"]')?.getAttribute('aria-label')).toBe('Loading Test widget');
    expect(host?.querySelector('[role="status"]')).not.toBeNull();
  });

  it('renders compact widget identity instead of a generic loading message', async () => {
    const { widget } = createTestWidget();
    await render(<WidgetLoadingFallback instance={instance} presentation="compact" region="bottom" widget={widget} />);

    expect(host?.textContent).toBe('Test widget');
    expect(host?.querySelector('[aria-busy="true"]')?.getAttribute('aria-label')).toBe('Loading Test widget');
    expect(host?.querySelector('.chakra-spinner')).not.toBeNull();
  });

  it('shows visible widget identity for inline regions', async () => {
    const { widget } = createTestWidget();
    await render(<WidgetLoadingFallback instance={instance} region="dialog" widget={widget} />);

    expect(host?.textContent).toBe('Test widget');
    expect(host?.querySelector('[role="status"]')?.getAttribute('aria-label')).toBe('Loading Test widget');
    expect(host?.querySelector('.chakra-spinner')).not.toBeNull();
  });
});

describe('widget intent preloading', () => {
  it('starts one cached implementation load from pointer and keyboard menu intent', async () => {
    const { loader, widget } = createTestWidget();
    await render(
      <WidgetEnableMenu
        groupLabel="Widgets"
        items={[
          {
            allowMultiple: false,
            icon: TestIcon,
            id: 'center:new:test',
            isEnabled: false,
            label: 'Test widget',
            typeId: 'test',
            widget,
          },
        ]}
        positioning={{ placement: 'bottom-end' }}
        trigger={{ kind: 'center' }}
        triggerLabel="Center widgets"
        onToggle={vi.fn()}
      />
    );

    await interact(() => host?.querySelector<HTMLButtonElement>('button[aria-label="Center widgets"]')?.click());
    const item = document.querySelector<HTMLElement>('[role="menuitemcheckbox"]');
    expect(item).not.toBeNull();

    await interact(() => item?.dispatchEvent(new PointerEvent('pointerover', { bubbles: true })));
    await interact(() => item?.focus());

    expect(loader).toHaveBeenCalledOnce();
  });

  it('does not load disabled widget implementations from pointer intent', async () => {
    const { loader, widget } = createTestWidget();
    widget.status = 'disabled';
    await render(
      <WidgetEnableMenu
        groupLabel="Widgets"
        items={[
          {
            allowMultiple: false,
            icon: TestIcon,
            id: 'center:new:test',
            isEnabled: false,
            label: 'Test widget',
            status: 'disabled',
            typeId: 'test',
            widget,
          },
        ]}
        positioning={{ placement: 'bottom-end' }}
        trigger={{ kind: 'center' }}
        triggerLabel="Center widgets"
        onToggle={vi.fn()}
      />
    );

    await interact(() => host?.querySelector<HTMLButtonElement>('button[aria-label="Center widgets"]')?.click());
    const item = document.querySelector<HTMLElement>('[role="menuitemcheckbox"]');
    expect(item).not.toBeNull();

    await interact(() => item?.dispatchEvent(new PointerEvent('pointerover', { bubbles: true })));

    expect(loader).not.toHaveBeenCalled();
  });
});
