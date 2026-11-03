import { ChakraProvider, createListCollection, Input } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it } from 'vitest';

import { Combobox } from './Combobox';
import { Field, useFieldLabelId } from './Field';
import { FeatureHintsProvider } from './hints';
import { openHintCard } from './hints/featureHint.testing';
import { ScrubberField } from './ScrubberField';
import { Select } from './Select';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: { hints: { steps: { heading: 'Steps', paragraphs: ['How many steps.'] } } } } },
});

const HINTS_ENABLED = { enabled: true, onDisable: null };
const STEP_COLLECTION = createListCollection({ items: [{ label: '30', value: '30' }] });
const STEP_VALUE = ['30'];
const SCHEDULER_OPTIONS = [{ label: 'Euler', value: 'euler' }];
const noop = () => undefined;
const LabelIdProbe = () => <output>{useFieldLabelId() ?? 'none'}</output>;

/** Where a label's text begins, whatever padding its element carries. */
const textLeft = (element: Element): number => {
  const range = document.createRange();
  range.selectNodeContents(element);
  return range.getBoundingClientRect().left;
};

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
  // Inset from the parked pointer, and narrow enough to leave room for a hint card beside the field.
  host.style.cssText = 'width:120px;margin:120px 0 0 8px';
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
   * Keep the hint trigger inside the label so its generated ID cannot replace the control's aria-labelledby
   * target.
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

  it('opens the card beside the field instead of over its control', async () => {
    const hosted = await render(true);
    const control = hosted.querySelector('input')!.getBoundingClientRect();
    const card = await openHintCard(hosted.querySelector('[data-part="trigger"]')!);

    expect(card.left).toBeGreaterThanOrEqual(control.right);
  });
});

describe('Field with its label inside the control', () => {
  it('keeps the hint on the label text inside the trigger, without replacing the id that names the trigger', async () => {
    host = document.createElement('div');
    host.style.cssText = 'width:200px;margin:120px 0 0 8px';
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <FeatureHintsProvider adapter={HINTS_ENABLED}>
              <Field hint="steps" id="steps-field" label="Steps" labelPlacement="inside">
                <Select collection={STEP_COLLECTION} value={STEP_VALUE} />
              </Field>
            </FeatureHintsProvider>
          </I18nextProvider>
        </ChakraProvider>
      );
    });

    const trigger = host.querySelector<HTMLElement>('[data-scope="select"][data-part="trigger"]')!;
    const hintTarget = trigger.querySelector<HTMLElement>('[data-scope="hover-card"][data-part="trigger"]')!;
    expect(trigger.getAttribute('aria-labelledby')).toBe('steps-field-label');
    expect(hintTarget.textContent).toBe('Steps');
    expect(hintTarget.parentElement?.id).toBe('steps-field-label');
    const card = await openHintCard(hintTarget);
    expect(card.left).toBeGreaterThanOrEqual(trigger.getBoundingClientRect().right);
  });

  it('starts its label where a stacked ScrubberField starts its own, in the same type', async () => {
    host = document.createElement('div');
    host.style.width = '240px';
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <ScrubberField label="Steps" max={100} min={1} step={1} value={30} onChange={noop} />
            <Field id="preset-field" label="Preset" labelPlacement="inside">
              <Select collection={STEP_COLLECTION} value={STEP_VALUE} />
            </Field>
            <Field id="scheduler-field" label="Scheduler" labelPlacement="inside">
              <Combobox options={SCHEDULER_OPTIONS} value="euler" onValueChange={noop} />
            </Field>
          </I18nextProvider>
        </ChakraProvider>
      );
    });

    const scrubberLabel = host.querySelector<HTMLElement>('[data-part="label"]')!;
    const insideLabels = [
      document.getElementById('preset-field-label')!,
      document.getElementById('scheduler-field-label')!,
    ];
    const scrubberStyle = getComputedStyle(scrubberLabel);
    for (const label of insideLabels) {
      expect(textLeft(label)).toBeCloseTo(textLeft(scrubberLabel), 0);
      const style = getComputedStyle(label);
      expect([style.fontSize, style.fontWeight, style.color, style.letterSpacing]).toEqual([
        scrubberStyle.fontSize,
        scrubberStyle.fontWeight,
        scrubberStyle.color,
        scrubberStyle.letterSpacing,
      ]);
    }
  });

  it('reports no rendered label id to a child that does not show the inside label', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Field id="probe-field" label="Probe" labelPlacement="inside">
            <LabelIdProbe />
          </Field>
          <Field id="outside-field" label="Outside">
            <LabelIdProbe />
          </Field>
        </ChakraProvider>
      );
    });

    expect([...host.querySelectorAll('output')].map((output) => output.textContent)).toEqual([
      'none',
      'outside-field-label',
    ]);
  });
});
