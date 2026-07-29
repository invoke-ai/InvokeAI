import { act } from 'react';
import { expect, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

type FinalComputedStyles = Readonly<Partial<Pick<CSSStyleDeclaration, 'backgroundColor' | 'color'>>>;

export const waitForComputedStyles = async (element: Element, expected: FinalComputedStyles) => {
  await vi.waitFor(() => {
    const styles = getComputedStyle(element);

    for (const property of Object.keys(expected) as (keyof FinalComputedStyles)[]) {
      expect(styles[property]).toBe(expected[property]);
    }
  });
};

export const captureRowInteractionStyles = async (probe: HTMLButtonElement, hoverBackgroundColor: string) => {
  await act(async () => {
    await focusWithKeyboard(probe);
    await userEvent.hover(probe);
  });
  await waitForComputedStyles(probe, { backgroundColor: hoverBackgroundColor });
  const expected = getInteractionStyles(probe);

  await act(async () => {
    await userEvent.unhover(probe);
  });

  return expected;
};

export const expectRowInteractionStylesToMatch = async (
  expected: ReturnType<typeof getInteractionStyles>,
  row: HTMLButtonElement,
  hoverBackgroundColor: string
) => {
  await act(async () => {
    await focusWithKeyboard(row);
    await userEvent.hover(row);
  });
  await waitForComputedStyles(row, { backgroundColor: hoverBackgroundColor });

  expect(getInteractionStyles(row)).toEqual(expected);
};

export const expectRowInteractionsToMatch = async (
  probe: HTMLButtonElement,
  row: HTMLButtonElement,
  hoverBackgroundColor: string
) => {
  const expected = await captureRowInteractionStyles(probe, hoverBackgroundColor);
  await expectRowInteractionStylesToMatch(expected, row, hoverBackgroundColor);
};

export const getXsButtonDensityStyles = (element: HTMLElement) => {
  const styles = getComputedStyle(element);

  return {
    borderBottomColor: styles.borderBottomColor,
    borderBottomStyle: styles.borderBottomStyle,
    borderBottomWidth: styles.borderBottomWidth,
    borderLeftColor: styles.borderLeftColor,
    borderLeftStyle: styles.borderLeftStyle,
    borderLeftWidth: styles.borderLeftWidth,
    borderRightColor: styles.borderRightColor,
    borderRightStyle: styles.borderRightStyle,
    borderRightWidth: styles.borderRightWidth,
    borderTopColor: styles.borderTopColor,
    borderTopStyle: styles.borderTopStyle,
    borderTopWidth: styles.borderTopWidth,
    fontSize: styles.fontSize,
    fontWeight: styles.fontWeight,
    lineHeight: styles.lineHeight,
    whiteSpace: styles.whiteSpace,
  };
};

export const matchElementWidth = (element: HTMLElement, reference: HTMLElement) => {
  element.style.width = `${reference.offsetWidth}px`;
};

export const getElementBoxMetrics = (element: HTMLElement) => ({
  clientHeight: element.clientHeight,
  clientWidth: element.clientWidth,
  offsetHeight: element.offsetHeight,
  offsetWidth: element.offsetWidth,
  scrollHeight: element.scrollHeight,
  scrollWidth: element.scrollWidth,
});

export const getRenderedLineCount = (element: HTMLElement) => {
  const range = document.createRange();
  range.selectNodeContents(element);
  const lineTops = new Set(
    [...range.getClientRects()].filter((rect) => rect.width > 0).map((rect) => Math.round(rect.top * 100) / 100)
  );

  return lineTops.size;
};

const focusWithKeyboard = async (element: HTMLButtonElement) => {
  for (let index = 0; index < 12; index += 1) {
    if (document.activeElement === element) {
      return;
    }
    await userEvent.tab();
  }

  throw new Error(`Could not focus ${element.title || element.textContent} with the keyboard`);
};

const getInteractionStyles = (element: HTMLElement) => {
  const styles = getComputedStyle(element);

  return {
    backgroundColor: styles.backgroundColor,
    borderRadius: styles.borderRadius,
    outline: styles.outline,
    outlineOffset: styles.outlineOffset,
    transitionDuration: styles.transitionDuration,
    transitionProperty: styles.transitionProperty,
    transitionTimingFunction: styles.transitionTimingFunction,
  };
};
