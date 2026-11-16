import type { ModelLoadInfo } from '@features/models';
import type { QueueActivity } from '@features/queue/contracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({
  activity: 'idle' as QueueActivity,
  modelLoads: [] as ModelLoadInfo[],
  remaining: 0,
}));

// The frame barrel reaches the whole workbench context; the chip itself is the platform control.
vi.mock('@workbench/widget-frame', async () => ({
  StatusWidgetChip: (await import('@platform/ui/StatusWidgetChip')).StatusWidgetChip,
}));
vi.mock('@workbench/queue-integration/useActiveQueueProgress', () => ({
  useQueueChromeActivity: () => ({
    activity: harness.activity,
    elsewhere: 0,
    modelLoads: harness.modelLoads,
    progress: undefined,
    remaining: harness.remaining,
    summary: { total: harness.remaining },
  }),
}));

import { QueueStatusWidgetView } from './QueueStatusWidgetView';

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const renderChip = async (): Promise<string> => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <QueueStatusWidgetView {...({ presentation: 'compact', region: 'bottom' } as WidgetViewProps)} />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
  return host!.textContent ?? '';
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  harness.activity = 'idle';
  harness.modelLoads = [];
  harness.remaining = 0;
});

describe('QueueStatusWidgetView', () => {
  it('names the loading model alongside the queued count', async () => {
    harness.activity = 'loading-model';
    harness.modelLoads = [{ label: 'FLUX.2 Klein 4B' }];
    harness.remaining = 2;

    expect(await renderChip()).toContain('Loading FLUX.2 Klein 4B · 2 queued');
  });

  it('says a model is loading without a blank name when none is reported yet', async () => {
    harness.activity = 'loading-model';
    harness.remaining = 1;

    const text = await renderChip();

    expect(text).toContain('Loading model · 1 queued');
    expect(text).not.toContain('widgets.queueStatus');
  });
});
