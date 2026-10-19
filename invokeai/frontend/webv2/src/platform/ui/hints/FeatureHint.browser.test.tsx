import { Box, ChakraProvider, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { FeatureHintsAdapter } from './hintsContext';

import { FeatureHint } from './FeatureHint';
import { movePointerAlong, openHintCard, parkPointer } from './featureHint.testing';
import { FEATURE_HINT_FIELD_PROPS } from './hintGeometry';
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

const HINTS_ON: FeatureHintsAdapter = { enabled: true, onDisable: null };
const FULL_WIDTH = { width: '100%' };

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// Reset the pointer after each test so a newly mounted trigger cannot open under the previous test's cursor.
afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  await parkPointer();
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
            {/* Inset the trigger from the initial pointer position so entering it emits pointerenter. */}
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

const waitForCard = async (timeoutMs = 5000): Promise<void> => {
  const deadline = Date.now() + timeoutMs;

  while (cardText() === '') {
    if (Date.now() > deadline) {
      throw new Error(`Timed out after ${timeoutMs}ms waiting for the hint card to open`);
    }

    await settle(16);
  }
};

const PANEL_CSS = 'position:fixed;top:120px;height:200px;overflow-y:auto';

/**
 * A hinted label above its control inside `host`: a scrolling panel when its CSS scrolls, like a docked widget
 * body. With `field`, the label and control sit in a marked 100px field.
 */
const renderHint = async (hostCss: string, { field = false, labels = ['CLIP skip'] } = {}) => {
  host = document.createElement('div');
  host.style.cssText = hostCss;
  document.body.append(host);
  root = createRoot(host);

  const content = (
    <>
      {labels.map((label) => (
        <FeatureHint key={label} hint="clipSkip">
          <Text display="inline-block" me="1">
            {label}
          </Text>
        </FeatureHint>
      ))}
      <input aria-label="CLIP skip value" style={FULL_WIDTH} />
    </>
  );

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <FeatureHintsProvider adapter={HINTS_ON}>
            {field ? (
              <Box {...FEATURE_HINT_FIELD_PROPS} w="100px">
                {content}
              </Box>
            ) : (
              content
            )}
          </FeatureHintsProvider>
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return {
    control: host.querySelector('input')!,
    panel: host,
    trigger: host.querySelector<HTMLElement>('[data-part="trigger"]')!,
    triggers: [...host.querySelectorAll<HTMLElement>('[data-part="trigger"]')],
  };
};

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
    await waitForCard();

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
    await waitForCard();

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
    await waitForCard();

    expect(cardText()).toContain('CLIP Skip');
    expect(cardText()).not.toContain("Don't show me these");
  });

  describe('placement', () => {
    it('sits beside its panel, level with the label, instead of over the controls in it', async () => {
      const { panel, trigger } = await renderHint(`${PANEL_CSS};width:80px;left:8px`);
      const card = await openHintCard(trigger);
      const label = trigger.getBoundingClientRect();

      expect(card.left).toBeGreaterThanOrEqual(panel.getBoundingClientRect().right);
      expect(card.top).toBeLessThan(label.bottom);
      expect(card.bottom).toBeGreaterThan(label.top);
    });

    it('flips to the other side of the panel when the viewport edge leaves no room', async () => {
      const { panel, trigger } = await renderHint(`${PANEL_CSS};width:80px;right:8px`);
      const card = await openHintCard(trigger);

      expect(card.right).toBeLessThanOrEqual(panel.getBoundingClientRect().left);
    });

    it('sits beside the field, on screen, when the panel fills the viewport', async () => {
      // A maximized floating window: no room beside the panel on either side.
      const { control, trigger } = await renderHint(`${PANEL_CSS};left:0;width:100vw`, { field: true });
      const card = await openHintCard(trigger);

      expect(card.left).toBeGreaterThanOrEqual(control.getBoundingClientRect().right);
      expect(card.right).toBeLessThanOrEqual(document.documentElement.clientWidth);
    });

    it('sits beside the marked field when no panel scrolls around it', async () => {
      const { control, trigger } = await renderHint('position:fixed;top:120px;left:8px', { field: true });
      const card = await openHintCard(trigger);

      expect(card.left).toBeGreaterThanOrEqual(control.getBoundingClientRect().right);
    });
  });

  describe('on the way to the card', () => {
    const LEFT_PANEL_CSS = `${PANEL_CSS};width:80px;left:8px`;
    const MOVE = { stepDelayMs: 20, steps: 4 };

    it('stays open while the pointer travels from the label to the link in the card', async () => {
      const { trigger } = await renderHint(LEFT_PANEL_CSS);
      const card = await openHintCard(trigger);
      const link = [...document.querySelectorAll('a')].find((anchor) => anchor.textContent === 'Learn more')!;
      const label = trigger.getBoundingClientRect();
      const to = link.getBoundingClientRect();
      const exit = { x: label.right - 2, y: label.top + label.height / 2 };
      // Short of the card and clear of its arrow, which points at the label and counts as the card.
      const pause = { x: card.left - 4, y: exit.y + 24 };

      // A pause that outlasts the close delay: only the corridor keeps the card open through it.
      await movePointerAlong(exit, pause, MOVE);
      await settle(400);
      await movePointerAlong(pause, { x: to.left + to.width / 2, y: to.top + to.height / 2 }, MOVE);
      await settle(500);

      expect(cardText()).toContain('Learn more');
      expect(document.elementFromPoint(to.left + to.width / 2, to.top + to.height / 2)?.closest('a')).toBe(link);
    });

    it('stays open through a rest on the arrow, which lies outside the card body', async () => {
      const { trigger } = await renderHint(LEFT_PANEL_CSS);
      const card = await openHintCard(trigger);
      const arrow = document.querySelector<HTMLElement>('[data-scope="hover-card"][data-part="arrow"]')!;
      const arrowRect = arrow.getBoundingClientRect();
      const label = trigger.getBoundingClientRect();
      const exit = { x: label.right - 2, y: label.top + label.height / 2 };
      // The arrow straddles the card's edge: its outer half is clear of the content rect.
      const outer = { x: arrowRect.left + 1, y: arrowRect.top + arrowRect.height / 2 };

      expect(outer.x).toBeLessThan(card.left);
      await movePointerAlong(exit, { x: outer.x - 6, y: outer.y }, MOVE);
      await act(() =>
        userEvent.hover(arrow, { position: { x: outer.x - arrowRect.left, y: outer.y - arrowRect.top } })
      );
      // Past the corridor's grace window and the close delay after it.
      await settle(1500);
      await act(() => userEvent.hover(document.querySelector('[data-scope="hover-card"][data-part="content"] h2')!));
      await settle(500);

      expect(cardText()).toContain('CLIP Skip');
    });

    it('closes at the ordinary delay when the pointer leaves with a button held, as a scrub does', async () => {
      const { trigger } = await renderHint(LEFT_PANEL_CSS);
      const card = await openHintCard(trigger);
      const label = trigger.getBoundingClientRect();
      const held = {
        bubbles: true,
        buttons: 1,
        clientX: label.right - 2,
        clientY: label.top + label.height / 2,
        pointerType: 'mouse',
      };

      // A scrubber taking pointer capture on press moves the pointer off the label with the button still down.
      await act(() => trigger.dispatchEvent(new PointerEvent('pointerdown', held)));
      expect(cardText()).toContain('CLIP Skip');
      await act(() => trigger.dispatchEvent(new PointerEvent('pointerout', { ...held, relatedTarget: document.body })));
      // A rightward drag heads toward the card; within the grace window, a corridor would keep the card open.
      for (const step of [1, 2, 3]) {
        await act(() =>
          document.dispatchEvent(new PointerEvent('pointermove', { ...held, clientX: held.clientX + step * 8 }))
        );
        await settle(100);
      }
      await settle(400);

      expect(card.left).toBeGreaterThan(held.clientX);
      expect(cardText()).toBe('');
    });

    it('closes once the pointer leaves the corridor sideways', async () => {
      const { trigger } = await renderHint(LEFT_PANEL_CSS);

      await openHintCard(trigger);

      const from = trigger.getBoundingClientRect();

      // Down the panel, away from the card beside it.
      await movePointerAlong(
        { x: from.left + from.width / 2, y: from.bottom - 2 },
        { x: from.left + from.width / 2, y: from.bottom + 80 },
        MOVE
      );
      await settle(600);

      expect(cardText()).toBe('');
    });

    it('closes after a grace window when the pointer rests on the way', async () => {
      const { trigger } = await renderHint(LEFT_PANEL_CSS);
      const card = await openHintCard(trigger);
      const label = trigger.getBoundingClientRect();
      const exit = { x: label.right - 2, y: label.top + label.height / 2 };

      await movePointerAlong(exit, { x: card.left - 4, y: exit.y + 24 }, MOVE);
      // Past the grace window and the close delay after it.
      await settle(1700);

      expect(cardText()).toBe('');
    });

    it('hands over to another hinted label on the way, so only one card is open', async () => {
      const {
        triggers: [first, second],
      } = await renderHint(LEFT_PANEL_CSS, { labels: ['CLIP', 'Skip'] });

      await openHintCard(first!);
      await act(() => userEvent.hover(second!));
      await expect.poll(() => second!.getAttribute('data-state'), { timeout: 3000 }).toBe('open');

      expect(document.querySelectorAll('[data-scope="hover-card"][data-part="content"]')).toHaveLength(1);
      expect(first!.getAttribute('data-state')).toBe('closed');
    });
  });
});
