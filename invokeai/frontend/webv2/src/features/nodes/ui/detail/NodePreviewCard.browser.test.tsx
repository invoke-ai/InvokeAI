import '@fontsource/inter/index.css';
import type { FieldInputTemplate, InvocationTemplate } from '@features/workflow/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { NodePreviewCard } from './NodePreviewCard';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const input: FieldInputTemplate = {
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'connection',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name: 'image',
  options: null,
  required: true,
  title: 'Image',
  type: { batch: false, cardinality: 'SINGLE', name: 'ImageField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
};
const template: InvocationTemplate = {
  category: 'image',
  classification: 'stable',
  description: '',
  inputs: { image: input },
  nodePack: 'invokeai',
  outputType: 'image_output',
  outputs: {
    width: {
      description: '',
      name: 'width',
      title: 'Width',
      type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
    },
  },
  tags: [],
  title: 'Resize Image',
  type: 'img_resize',
  useCache: true,
  version: '1.0.0',
};

// A title and an output title split next to a space: the head keeps its trailing space.
const splitTemplate: InvocationTemplate = {
  ...template,
  outputs: { width: { ...template.outputs.width!, title: 'Image to Latents' } },
  title: 'Latents to Image',
};

/** The middle of a label's lowercase letters: a 1ex inline box standing on its baseline spans its x-height. */
const letterBodyCentre = (text: HTMLElement) => {
  const probe = document.createElement('span');
  probe.style.cssText = 'display: inline-block; height: 1ex; vertical-align: baseline; width: 0';
  text.append(probe);
  const box = probe.getBoundingClientRect();
  probe.remove();
  return box.top + box.height / 2;
};

describe('NodePreviewCard', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.width = '320px';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('keeps the space where its title or an output title is split, as an untrimmed label draws it', async () => {
    await document.fonts.load('600 12px Inter');
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <NodePreviewCard template={splitTemplate} />
        </ChakraProvider>
      )
    );

    for (const text of ['Latents to Image', 'Image to Latents']) {
      const label = host.querySelector<HTMLElement>(`[data-full-text="${text}"]`)!;
      const [head, tail] = [...label.children] as HTMLElement[];
      expect(getComputedStyle(head!).whiteSpace).toBe('pre');
      // The same label untrimmed, beside it in the same font: its spans hold their own white-space.
      const reference = label.cloneNode(true) as HTMLElement;
      reference.style.cssText = 'position: absolute; visibility: hidden; overflow: visible';
      for (const span of reference.children as HTMLCollectionOf<HTMLElement>) {
        span.style.cssText = 'text-box: normal; white-space: pre; flex-shrink: 0';
      }
      label.parentElement!.append(reference);
      const width = [...reference.children].reduce((sum, span) => sum + span.getBoundingClientRect().width, 0);
      reference.remove();
      expect(head!.getBoundingClientRect().width + tail!.getBoundingClientRect().width).toBeCloseTo(width, 0);
    }
  });

  it('centres each port dot on its label’s letters, as a node’s field rows do', async () => {
    await document.fonts.load('400 10px Inter');
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <NodePreviewCard template={template} />
        </ChakraProvider>
      )
    );

    for (const [side, title] of [
      ['left', 'Image'],
      ['right', 'Width'],
    ] as const) {
      const dot = host.querySelector<HTMLElement>(`[data-node-port-dot="${side}"]`)!;
      const label = [...dot.parentElement!.querySelectorAll<HTMLElement>('p, span')].find((element) =>
        [...element.childNodes].some((child) => child.nodeType === Node.TEXT_NODE && child.textContent === title)
      )!;
      const box = dot.getBoundingClientRect();
      expect(Math.abs(box.top + box.height / 2 - letterBodyCentre(label))).toBeLessThanOrEqual(0.5);
    }
  });
});
