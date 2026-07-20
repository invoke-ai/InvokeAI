import type { FieldInputTemplate } from '@features/workflow/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowFieldInput } from './WorkflowFieldInput';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const TEXTAREA_TEMPLATE = {
  name: 'prompt',
  title: 'Prompt',
  type: { name: 'StringField' },
  uiComponent: 'textarea',
} as unknown as FieldInputTemplate;

describe('WorkflowFieldInput textarea', () => {
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

  it('uses the accessible unbounded resizable textarea for prompt-like string fields', async () => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <WorkflowFieldInput template={TEXTAREA_TEMPLATE} value="hello" onChange={vi.fn()} />
        </ChakraProvider>
      );
    });

    const textarea = host.querySelector<HTMLTextAreaElement>('textarea')!;
    const handle = host.querySelector<HTMLElement>('[role="separator"]')!;

    expect(getComputedStyle(textarea).height).toBe('96px');
    expect(getComputedStyle(textarea).fontFamily).toContain('monospace');
    expect(handle.getAttribute('aria-label')).toBe('Resize Prompt');
    expect(handle.getAttribute('aria-valuemin')).toBe('56');
    expect(handle.hasAttribute('aria-valuemax')).toBe(false);

    await act(() => handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));
    expect(getComputedStyle(textarea).height).toBe('108px');
  });
});
