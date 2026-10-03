import {
  Button,
  ChakraProvider,
  Combobox,
  createListCollection,
  Input,
  NumberInput,
  SegmentGroup,
  Select,
} from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { CONTROL_HEIGHT_PX, type ControlSize } from './scale';

const OPTIONS = createListCollection({ items: [{ label: 'One', value: 'one' }] });

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const renderControls = async (size?: ControlSize): Promise<Record<string, HTMLElement>> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Button data-testid="button" size={size}>
          Reference
        </Button>
        <Input aria-label="Input" data-testid="input" size={size} />
        <NumberInput.Root size={size}>
          <NumberInput.Input aria-label="Number" data-testid="number" />
        </NumberInput.Root>
        <SegmentGroup.Root data-testid="segments" size={size} value="a">
          <SegmentGroup.Indicator />
          {['a', 'b'].map((value) => (
            <SegmentGroup.Item key={value} value={value}>
              <SegmentGroup.ItemHiddenInput />
              <SegmentGroup.ItemText>{value}</SegmentGroup.ItemText>
            </SegmentGroup.Item>
          ))}
        </SegmentGroup.Root>
        <Select.Root collection={OPTIONS} size={size}>
          <Select.Control>
            <Select.Trigger aria-label="Select" data-testid="select">
              <Select.ValueText placeholder="Pick" />
            </Select.Trigger>
          </Select.Control>
        </Select.Root>
        <Combobox.Root collection={OPTIONS} size={size}>
          <Combobox.Control>
            <Combobox.Input aria-label="Combobox" data-testid="combobox" />
          </Combobox.Control>
        </Combobox.Root>
      </ChakraProvider>
    );
  });

  return Object.fromEntries(
    ['button', 'input', 'number', 'segments', 'select', 'combobox'].map((id) => [
      id,
      host!.querySelector<HTMLElement>(`[data-testid="${id}"]`)!,
    ])
  );
};

describe('control heights', () => {
  // Same-named control sizes share the scale's outer height, so mixed rows of controls align.
  (Object.entries(CONTROL_HEIGHT_PX) as [ControlSize, number][]).forEach(([size, height]) => {
    it(`renders ${size} controls at ${height}px`, async () => {
      const controls = await renderControls(size);

      for (const [id, element] of Object.entries(controls)) {
        expect(element.getBoundingClientRect().height, id).toBeCloseTo(height, 1);
      }
    });
  });

  it('renders controls that name no size at md', async () => {
    const controls = await renderControls();

    for (const [id, element] of Object.entries(controls)) {
      expect(element.getBoundingClientRect().height, id).toBeCloseTo(CONTROL_HEIGHT_PX.md, 1);
    }
  });
});
