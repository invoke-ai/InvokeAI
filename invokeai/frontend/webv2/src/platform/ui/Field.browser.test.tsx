import { ChakraProvider, Input } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it } from 'vitest';

import { Field } from './Field';
import { FeatureHintsProvider } from './hints';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: { hints: { steps: { heading: 'Steps', paragraphs: ['How many steps.'] } } } } },
});

const HINTS_ENABLED = { enabled: true, onDisable: null };

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const render = async (hinted: boolean) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <FeatureHintsProvider adapter={HINTS_ENABLED}>
            <Field hint={hinted ? 'steps' : undefined} id="steps-field" label="Steps">
              <Input />
            </Field>
          </FeatureHintsProvider>
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return host;
};

describe('Field with a hint', () => {
  /**
   * The hover-card trigger stamps its own `id` on whatever element it wraps, so
   * making the `<label>` itself the trigger replaced the Field-derived id that
   * controls reference through `aria-labelledby` — axe reported every hinted
   * Select as having no accessible name. The trigger must sit *inside* the label.
   */
  it('keeps the Field-derived label id when a hint is attached', async () => {
    const hosted = await render(true);
    const label = hosted.querySelector('label');

    expect(label?.id).toBe('steps-field-label');
    expect(hosted.querySelector('[data-part="trigger"]')?.id).not.toBe('steps-field-label');
  });

  it('renders the same label id and text with and without a hint', async () => {
    const hinted = await render(true);
    const hintedLabel = hinted.querySelector('label');

    expect(hintedLabel?.textContent).toBe('Steps');
    await act(() => root?.unmount());
    host?.remove();

    const plain = await render(false);
    const plainLabel = plain.querySelector('label');

    expect(plainLabel?.id).toBe(hintedLabel?.id);
    expect(plainLabel?.textContent).toBe('Steps');
    expect(plain.querySelector('[data-part="trigger"]')).toBeNull();
  });

  it('makes the label text the hover target', async () => {
    const hosted = await render(true);

    expect(hosted.querySelector('label [data-part="trigger"]')?.textContent).toBe('Steps');
  });
});
