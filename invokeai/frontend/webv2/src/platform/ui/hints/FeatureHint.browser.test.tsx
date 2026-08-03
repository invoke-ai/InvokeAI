import { ChakraProvider, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { FeatureHintsAdapter } from './hintsContext';

import { FeatureHint } from './FeatureHint';
import { FeatureHintsProvider } from './hintsContext';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { dontShowMeThese: "Don't show me these", learnMore: 'Learn more' },
        hints: {
          clipSkip: {
            heading: 'CLIP Skip',
            paragraphs: ['How many layers of the CLIP model to skip.', 'Some models suit CLIP Skip better.'],
          },
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const settle = (ms: number): Promise<void> =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, ms);
    });
  });

const render = async (adapter: FeatureHintsAdapter) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <FeatureHintsProvider adapter={adapter}>
            {/*
              Inset and shrink-wrapped on purpose: the mouse starts at the
              viewport origin, so a full-width trigger at (0,0) would already
              contain the pointer and `pointerenter` would never fire.
            */}
            <FeatureHint hint="clipSkip">
              <Text display="inline-block" m="120px">
                CLIP skip
              </Text>
            </FeatureHint>
          </FeatureHintsProvider>
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return host.querySelector<HTMLElement>('[data-part="trigger"]') ?? (host.firstElementChild as HTMLElement);
};

/** The card portals out of the host, so assert against the whole document. */
const cardText = (): string =>
  document.querySelector('[data-scope="hover-card"][data-part="content"]')?.textContent ?? '';

describe('FeatureHint', () => {
  it('renders the child untouched and mounts no popper while hints are off', async () => {
    const trigger = await render({ enabled: false, onDisable: vi.fn() });

    expect(trigger.textContent).toBe('CLIP skip');
    expect(trigger.getAttribute('data-scope')).toBeNull();
    expect(document.querySelector('[data-scope="hover-card"]')).toBeNull();
  });

  it('keeps the trigger out of the tab order so wrapping labels adds no tab stops', async () => {
    const trigger = await render({ enabled: true, onDisable: vi.fn() });

    expect(trigger.getAttribute('data-scope')).toBe('hover-card');
    expect(trigger.hasAttribute('tabindex')).toBe(false);
    expect(trigger.getAttribute('role')).toBeNull();
  });

  it('opens the card with its heading and paragraphs on hover', async () => {
    const trigger = await render({ enabled: true, onDisable: vi.fn() });

    await act(async () => {
      await userEvent.hover(trigger);
    });
    // Guard against a false pass from an instantly-open card.
    expect(cardText()).toBe('');
    await settle(900);

    expect(cardText()).toContain('CLIP Skip');
    expect(cardText()).toContain('How many layers of the CLIP model to skip.');
    expect(cardText()).toContain('Some models suit CLIP Skip better.');
  });

  it('turns hints off from the card', async () => {
    const onDisable = vi.fn();
    const trigger = await render({ enabled: true, onDisable });

    await act(async () => {
      await userEvent.hover(trigger);
    });
    await settle(900);

    const dismiss = [...document.querySelectorAll('button')].find(
      (button) => button.textContent === "Don't show me these"
    );

    expect(dismiss).toBeDefined();
    await act(() => {
      dismiss?.click();
    });

    expect(onDisable).toHaveBeenCalledTimes(1);
  });

  it('omits the dismiss action when the host cannot persist preferences', async () => {
    const trigger = await render({ enabled: true, onDisable: null });

    await act(async () => {
      await userEvent.hover(trigger);
    });
    await settle(900);

    expect(cardText()).toContain('CLIP Skip');
    expect(cardText()).not.toContain("Don't show me these");
  });
});
