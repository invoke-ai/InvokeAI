import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { LinkifiedText } from './LinkifiedText';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('LinkifiedText', () => {
  it('renders URLs as safe new-tab links and leaves the rest of the text as written', async () => {
    const content = 'Get the model at https://example.com/models?x=1. <b>not html</b>';

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <LinkifiedText fontSize="xs">{content}</LinkifiedText>
        </ChakraProvider>
      );
    });

    const anchors = host.querySelectorAll('a');
    expect(anchors).toHaveLength(1);
    expect(anchors[0]!.getAttribute('href')).toBe('https://example.com/models?x=1');
    expect(anchors[0]!.getAttribute('target')).toBe('_blank');
    expect(anchors[0]!.getAttribute('rel')).toBe('noopener noreferrer');
    expect(anchors[0]!.textContent).toBe('https://example.com/models?x=1');
    expect(host.querySelector('b')).toBeNull();
    expect(host.textContent).toBe(content);
  });
});
