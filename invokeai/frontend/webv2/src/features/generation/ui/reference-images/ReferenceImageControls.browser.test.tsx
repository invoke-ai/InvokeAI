import type { GenerateReferenceImageConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { CLIP_VISION_MODELS } from './referenceImageConfig';
import { IPAdapterControls } from './ReferenceImageControls';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: { translation: { widgets: { generate: { activeSteps: 'Active steps', mode: 'Mode', weight: 'Weight' } } } },
  },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type IPAdapterConfig = Extract<GenerateReferenceImageConfig, { type: 'ip_adapter' }>;

const CONFIG: IPAdapterConfig = {
  beginEndStepPct: [0, 1],
  clipVisionModel: CLIP_VISION_MODELS[0]!,
  image: null,
  method: 'full',
  model: null,
  type: 'ip_adapter',
  weight: 1,
};

/** Track inset the scrubber keeps from its frame edge (see ScrubberField). */
const TRACK_INSET_PX = 10;

let host: HTMLDivElement;
let root: Root;
const onChange = vi.fn();

/** Mirrors the real card: each commit lands back on the rendered config. */
const Harness = () => {
  const [config, setConfig] = useState<IPAdapterConfig>(CONFIG);
  const handleChange = useCallback((next: GenerateReferenceImageConfig) => {
    onChange(next);
    setConfig(next as IPAdapterConfig);
  }, []);

  return <IPAdapterControls config={config} disabled={false} onChange={handleChange} />;
};

beforeEach(() => {
  onChange.mockClear();
  host = document.createElement('div');
  host.style.width = '320px';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const pointer = (target: EventTarget, type: string, init: PointerEventInit) =>
  act(() => target.dispatchEvent(new PointerEvent(type, { bubbles: true, button: 0, ...init })));

describe('IPAdapterControls weight', () => {
  it('commits every step of a drag, including the one back to where the drag started', async () => {
    await act(() =>
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <Harness />
          </ChakraProvider>
        </I18nextProvider>
      )
    );

    const frame = host.querySelector<HTMLElement>('[data-scope="scrubber"]')!;
    const rect = frame.getBoundingClientRect();
    const trackWidth = rect.width - TRACK_INSET_PX * 2;
    // 10% of the 0..2 track is 0.2 of weight.
    const x = (fraction: number) => rect.left + TRACK_INSET_PX + fraction * trackWidth;

    await pointer(frame, 'pointerdown', { clientX: x(0.5) });
    await pointer(window, 'pointermove', { clientX: x(0.6) });

    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ weight: 1.2 }));

    await pointer(window, 'pointermove', { clientX: x(0.5) });
    await pointer(window, 'pointerup', { clientX: x(0.5) });

    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ weight: 1 }));
    expect(host.querySelector('[role="slider"]')?.getAttribute('aria-valuenow')).toBe('1');
  });
});
