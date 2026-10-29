import {
  Box,
  ChakraProvider,
  Combobox,
  createListCollection,
  Input,
  NumberInput,
  Select,
  Text,
  Textarea,
} from '@chakra-ui/react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { applyThemeAdjustmentsToRoot } from './adjustments';
import { applyThemeToRoot } from './applyTheme';
import { system } from './system';
import { THEMES_BY_ID } from './themes';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const OPTIONS = createListCollection({ items: [{ label: 'One', value: 'one' }] });

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  applyThemeAdjustmentsToRoot(THEMES_BY_ID.classic, undefined, false);
  delete document.documentElement.dataset.highContrast;
  applyThemeToRoot('classic');
  host = null;
  root = null;
});

describe('per-theme appearance adjustments', () => {
  it('changes native input surfaces and restores their original colors on reset or high contrast', async () => {
    applyThemeToRoot('classic');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Input aria-label="Name" data-testid="input" />
          <NumberInput.Root>
            <NumberInput.Input aria-label="Count" data-testid="number" />
          </NumberInput.Root>
          <Textarea aria-label="Description" data-testid="textarea" />
          <Select.Root collection={OPTIONS}>
            <Select.Control>
              <Select.Trigger aria-label="Select" data-testid="select">
                <Select.ValueText placeholder="Pick" />
              </Select.Trigger>
            </Select.Control>
          </Select.Root>
          <Combobox.Root collection={OPTIONS}>
            <Combobox.Control>
              <Combobox.Input aria-label="Combobox" data-testid="combobox" />
            </Combobox.Control>
          </Combobox.Root>
          <Text color="fg" data-testid="text">
            Example
          </Text>
          <Box bg="bg.subtle" data-testid="surface">
            <Text color="fg.muted" data-testid="muted">
              Muted
            </Text>
          </Box>
          <Box bg="bg.panel" data-testid="panel" />
          <Box bg="brand.solid" color="brand.contrast" data-testid="brand">
            Invoke
          </Box>
          <Box bg="accent.solid" color="accent.contrast" data-testid="accent">
            Selection
          </Box>
        </ChakraProvider>
      );
    });

    const input = host.querySelector<HTMLElement>('[data-testid="input"]')!;
    const number = host.querySelector<HTMLElement>('[data-testid="number"]')!;
    const textarea = host.querySelector<HTMLElement>('[data-testid="textarea"]')!;
    const select = host.querySelector<HTMLElement>('[data-testid="select"]')!;
    const combobox = host.querySelector<HTMLElement>('[data-testid="combobox"]')!;
    const text = host.querySelector<HTMLElement>('[data-testid="text"]')!;
    const surface = host.querySelector<HTMLElement>('[data-testid="surface"]')!;
    const muted = host.querySelector<HTMLElement>('[data-testid="muted"]')!;
    const panel = host.querySelector<HTMLElement>('[data-testid="panel"]')!;
    const brand = host.querySelector<HTMLElement>('[data-testid="brand"]')!;
    const accent = host.querySelector<HTMLElement>('[data-testid="accent"]')!;
    const original = {
      background: getComputedStyle(input).backgroundColor,
      border: getComputedStyle(input).borderTopColor,
      text: getComputedStyle(text).color,
      surface: getComputedStyle(surface).backgroundColor,
      muted: getComputedStyle(muted).color,
      panel: getComputedStyle(panel).backgroundColor,
      brand: getComputedStyle(brand).backgroundColor,
      brandText: getComputedStyle(brand).color,
      accent: getComputedStyle(accent).backgroundColor,
      accentText: getComputedStyle(accent).color,
    };
    input.setAttribute('aria-invalid', 'true');
    const invalidBorder = getComputedStyle(input).borderTopColor;
    input.removeAttribute('aria-invalid');

    applyThemeAdjustmentsToRoot(
      THEMES_BY_ID.classic,
      {
        backgroundLightness: 65,
        panelLightness: 70,
        inputLightness: 65,
        inputBorderLightness: 75,
        textLightness: 0,
        brandHex: '#123456',
        accentHex: '#eeeeee',
      },
      false
    );
    expect(getComputedStyle(input).backgroundColor).not.toBe(original.background);
    expect(getComputedStyle(number).backgroundColor).toBe(getComputedStyle(input).backgroundColor);
    expect(getComputedStyle(textarea).backgroundColor).toBe(getComputedStyle(input).backgroundColor);
    expect(getComputedStyle(select).backgroundColor).toBe(getComputedStyle(input).backgroundColor);
    expect(getComputedStyle(combobox).backgroundColor).toBe(getComputedStyle(input).backgroundColor);
    expect(getComputedStyle(input).borderTopColor).not.toBe(original.border);
    expect(getComputedStyle(text).color).not.toBe(original.text);
    expect(getComputedStyle(surface).backgroundColor).not.toBe(original.surface);
    expect(getComputedStyle(muted).color).not.toBe(original.muted);
    expect(getComputedStyle(panel).backgroundColor).not.toBe(original.panel);
    expect(getComputedStyle(brand).backgroundColor).not.toBe(original.brand);
    expect(getComputedStyle(brand).color).not.toBe(original.brandText);
    expect(getComputedStyle(accent).backgroundColor).not.toBe(original.accent);
    expect(getComputedStyle(accent).color).not.toBe(original.accentText);
    input.setAttribute('aria-invalid', 'true');
    await expect.poll(() => getComputedStyle(input).borderTopColor).toBe(invalidBorder);
    input.removeAttribute('aria-invalid');

    applyThemeAdjustmentsToRoot(THEMES_BY_ID.classic, undefined, false);
    expect(getComputedStyle(input).backgroundColor).toBe(original.background);
    await expect.poll(() => getComputedStyle(input).borderTopColor).toBe(original.border);
    expect(getComputedStyle(text).color).toBe(original.text);
    expect(getComputedStyle(surface).backgroundColor).toBe(original.surface);
    expect(getComputedStyle(muted).color).toBe(original.muted);
    expect(getComputedStyle(panel).backgroundColor).toBe(original.panel);
    expect(getComputedStyle(brand).backgroundColor).toBe(original.brand);
    expect(getComputedStyle(brand).color).toBe(original.brandText);
    expect(getComputedStyle(accent).backgroundColor).toBe(original.accent);
    expect(getComputedStyle(accent).color).toBe(original.accentText);

    applyThemeAdjustmentsToRoot(THEMES_BY_ID.classic, { inputLightness: 65, textLightness: 0 }, false);
    applyThemeAdjustmentsToRoot(THEMES_BY_ID.classic, { inputLightness: 65, textLightness: 0 }, true);
    expect(getComputedStyle(input).backgroundColor).toBe(original.background);
    expect(getComputedStyle(text).color).toBe(original.text);
  });
});
