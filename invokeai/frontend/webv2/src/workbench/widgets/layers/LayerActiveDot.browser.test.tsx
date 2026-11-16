import type { RowProps } from '@platform/ui/Row';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { Row } from '@platform/ui/Row';
import { applyThemeToRoot } from '@theme/applyTheme';
import { compositeColors, getContrastRatio } from '@theme/contrastRatio.testing';
import { system } from '@theme/system';
import { THEMES } from '@theme/themes';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { LayerActiveDot } from './LayerActiveDot';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  // Read settled colours: reduced motion shortens the transitions to a millisecond.
  document.documentElement.dataset.reduceMotion = 'true';
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  delete document.documentElement.dataset.reduceMotion;
  applyThemeToRoot('classic');
  host = null;
  root = null;
});

const noop = (): void => undefined;

/** Every fill a layer row can rest on: at rest, pointed, selected, primary, and the muted ancestor tone. */
const FILLS: readonly { id: string; active?: RowProps['active']; pointed?: boolean }[] = [
  { id: 'rest' },
  { id: 'hover', pointed: true },
  { id: 'selected', active: 'selected' },
  { id: 'primary', active: 'emphasized' },
  { id: 'muted', active: 'muted' },
];

const renderDots = async (checked: boolean, gatedReason: string | null = null) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Box bg="bg.panel" data-testid="panel" p="2">
          {FILLS.map(({ active, id, pointed }) => (
            <Row key={id} active={active} data-hover={pointed || undefined} data-testid={id} h="10" px="2">
              <LayerActiveDot
                checked={checked}
                disabled={false}
                gatedReason={gatedReason}
                label="Use in generation"
                tooltip="Use in generation"
                onCheckedChange={noop}
              />
            </Row>
          ))}
        </Box>
      </ChakraProvider>
    );
  });
  const panel = getComputedStyle(host.querySelector('[data-testid="panel"]')!).backgroundColor;
  return (id: string) => {
    const row = host!.querySelector<HTMLElement>(`[data-testid="${id}"]`)!;
    const fill = compositeColors(getComputedStyle(row).backgroundColor, panel);
    const ring = getComputedStyle(row.querySelector('button')!, '::before');
    return { fill, ring };
  };
};

describe('LayerActiveDot', () => {
  // A dot that is off is the faintest state of an on/off control: its ring stays a visible boundary (WCAG 1.4.11) on
  // every fill a row can take, in every theme, while staying a solid ring unlike the gated dashed one.
  it.each(THEMES.map((theme) => [theme.id, theme.colorScheme] as const))(
    'draws the off ring at 3:1 or more against every row fill in the %s (%s) theme',
    async (themeId) => {
      applyThemeToRoot(themeId);
      const measure = await renderDots(false);
      for (const { id } of FILLS) {
        const { fill, ring } = measure(id);
        expect(ring.borderTopStyle, id).toBe('solid');
        expect(ring.backgroundColor, id).toBe('rgba(0, 0, 0, 0)');
        expect(getContrastRatio(ring.borderTopColor, fill, 1), `${themeId} ${id}`).toBeGreaterThanOrEqual(3);
      }
    }
  );

  it('keeps the off ring apart from a dot that is on and from the dashed gated ring', async () => {
    const off = (await renderDots(false))('rest').ring;
    const offRing = { color: off.borderTopColor, style: off.borderTopStyle };
    await act(() => root?.unmount());
    host?.remove();
    const on = (await renderDots(true))('rest').ring;
    expect(on.backgroundColor).not.toBe('rgba(0, 0, 0, 0)');
    expect(on.borderTopColor).not.toBe(offRing.color);
    await act(() => root?.unmount());
    host?.remove();
    const gated = (await renderDots(true, 'Left out of generation by a group'))('rest').ring;
    expect(gated.borderTopStyle).toBe('dashed');
    expect(offRing.style).toBe('solid');
  });
});
