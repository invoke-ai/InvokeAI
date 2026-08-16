import { Box, ChakraProvider, DataList } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const renderDataList = async (): Promise<{ reference: HTMLElement; rows: HTMLElement[] }> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Box borderColor="border.subtle" borderWidth="1px" data-testid="reference" />
        <DataList.Root gap="1.5" orientation="horizontal" size="sm">
          {['Seed', 'Steps', 'Prompt'].map((label) => (
            <DataList.Item key={label} data-testid="row">
              <DataList.ItemLabel>{label}</DataList.ItemLabel>
              <DataList.ItemValue>value</DataList.ItemValue>
            </DataList.Item>
          ))}
        </DataList.Root>
      </ChakraProvider>
    );
  });

  return {
    reference: host.querySelector<HTMLElement>('[data-testid="reference"]')!,
    rows: [...host.querySelectorAll<HTMLElement>('[data-testid="row"]')],
  };
};

describe('dataList slot recipe', () => {
  it('draws a hairline divider above every row but the first', async () => {
    const { rows } = await renderDataList();
    const [first, second, third] = rows;

    expect(getComputedStyle(first).borderTopWidth).toBe('0px');
    expect(getComputedStyle(second).borderTopWidth).toBe('1px');
    expect(getComputedStyle(third).borderTopWidth).toBe('1px');
  });

  it('uses the subtle border token, not the inherited text color', async () => {
    const { reference, rows } = await renderDataList();

    // A failed token lookup would fall back to `currentColor`; comparing
    // against a border.subtle reference catches that regression.
    expect(getComputedStyle(rows[1]).borderTopColor).toBe(getComputedStyle(reference).borderTopColor);
    expect(getComputedStyle(rows[1]).borderTopColor).not.toBe(getComputedStyle(rows[1]).color);
  });

  it('pads divided rows so the line sits centered in the 1.5-unit row gap', async () => {
    const { rows } = await renderDataList();

    expect(getComputedStyle(rows[0]).paddingTop).toBe('0px');
    expect(getComputedStyle(rows[1]).paddingTop).toBe('6px');
  });
});
