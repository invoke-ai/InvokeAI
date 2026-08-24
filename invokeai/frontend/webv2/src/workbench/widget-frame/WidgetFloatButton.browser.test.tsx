/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { RegisteredWidget, WidgetManifest, WidgetTypeId, WorkbenchRegion } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import i18next from 'i18next';
import { MapIcon } from 'lucide-react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * The float control replaced a menu item, so the conditions that used to hide
 * that item now decide whether a button renders at all: a widget that may not
 * float, and the last center view — floating it out would leave the work
 * surface empty. Docking is covered by the floating window's own chrome.
 */

const floatMocks = vi.hoisted(() => ({
  centerInstanceIds: ['image-map-instance'] as string[],
  float: vi.fn(),
  flushWorkbenchDrafts: vi.fn(),
}));

vi.mock('@platform/react/draftRegistry', () => ({
  flushWorkbenchDrafts: floatMocks.flushWorkbenchDrafts,
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
    selector({
      id: 'project-1',
      widgetInstances: Object.fromEntries(
        floatMocks.centerInstanceIds.map((instanceId) => [
          instanceId,
          { createdAt: 0, id: instanceId, state: { values: {} }, typeId: 'image-map' },
        ])
      ),
      widgetRegions: {
        center: { activeInstanceId: floatMocks.centerInstanceIds[0], instanceIds: floatMocks.centerInstanceIds },
      },
    }),
  useWorkbenchCommands: () => ({ widgets: { float: floatMocks.float } }),
}));

vi.mock('@workbench/WorkbenchWidgetRegistryContext', () => ({
  useWorkbenchWidgetRegistry: () => ({
    getWidgetById: (typeId: WidgetTypeId) =>
      ({ manifest: { id: typeId }, status: 'enabled' }) as unknown as RegisteredWidget,
  }),
}));

import { WidgetFloatButton } from './WidgetFrames';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: { en: { translation: { widgets: { floating: { floatWindow: 'Float Window' } } } } },
});

const manifest = (allowFloating: boolean): WidgetManifest =>
  ({
    allowFloating,
    allowedRegions: ['center', 'left', 'right'],
    failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    icon: MapIcon,
    id: 'image-map',
    label: () => 'Image Map',
    load: () => Promise.resolve({ view: () => null }),
    version: 1,
  }) as unknown as WidgetManifest;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderButton = async (region: WorkbenchRegion, allowFloating = true) => {
  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <WidgetFloatButton instanceId="image-map-instance" manifest={manifest(allowFloating)} region={region} />
        </ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });

  return host?.querySelector<HTMLButtonElement>('button[aria-label="Float Window"]') ?? null;
};

beforeEach(() => {
  floatMocks.centerInstanceIds = ['image-map-instance'];
  floatMocks.float.mockClear();
  floatMocks.flushWorkbenchDrafts.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('WidgetFloatButton', () => {
  it('floats the instance from a docked panel, flushing drafts first', async () => {
    const button = await renderButton('right');

    expect(button).not.toBeNull();

    await act(async () => {
      button?.click();
      await Promise.resolve();
    });

    expect(floatMocks.flushWorkbenchDrafts).toHaveBeenCalled();
    expect(floatMocks.float).toHaveBeenCalledWith('image-map-instance');
    expect(floatMocks.flushWorkbenchDrafts.mock.invocationCallOrder[0]).toBeLessThan(
      floatMocks.float.mock.invocationCallOrder[0]
    );
  });

  it('renders nothing for a widget its manifest does not allow to float', async () => {
    expect(await renderButton('right', false)).toBeNull();
  });

  it('renders nothing for the last enabled center view', async () => {
    expect(await renderButton('center')).toBeNull();
  });

  it('renders for a center view that is not the last one', async () => {
    floatMocks.centerInstanceIds = ['image-map-instance', 'canvas-instance'];

    expect(await renderButton('center')).not.toBeNull();
  });
});
