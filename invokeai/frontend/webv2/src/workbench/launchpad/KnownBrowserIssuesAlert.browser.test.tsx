import type * as knownBrowserIssuesModule from '@platform/browser/knownBrowserIssues';

import { ChakraProvider } from '@chakra-ui/react';
import { KNOWN_BROWSER_ISSUES } from '@platform/browser/knownBrowserIssues';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const detectKnownBrowserIssuesMock = vi.hoisted(() => vi.fn());

vi.mock('@platform/browser/knownBrowserIssues', async (importOriginal) => ({
  ...(await importOriginal<typeof knownBrowserIssuesModule>()),
  detectKnownBrowserIssues: detectKnownBrowserIssuesMock,
}));

const translations: Record<string, string> = {
  'launchpad.browserIssues.canvasReadbackIntegrity.description':
    'Your browser is modifying or blocking canvas pixel data. This can make later strokes noisy or pixelated.',
  'launchpad.browserIssues.canvasReadbackIntegrity.genericWorkaround':
    'If you use another privacy browser or canvas-protection extension, allow unmodified canvas access for this site, then reload Invoke.',
  'launchpad.browserIssues.canvasReadbackIntegrity.heliumWorkaround':
    'Open the setting below, disable “[Helium Noise] Canvas pixel noising”, then relaunch Helium.',
  'launchpad.browserIssues.canvasReadbackIntegrity.title': 'Canvas protection may corrupt painted strokes',
  'launchpad.browserIssues.copyFailed': 'Could not copy browser setting',
  'launchpad.browserIssues.copySetting': 'Copy browser setting',
  'launchpad.browserIssues.settingCopied': 'Browser setting copied',
};

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => translations[key] ?? key,
  }),
}));

import { KnownBrowserIssuesAlert } from './KnownBrowserIssuesAlert';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let originalClipboardDescriptor: PropertyDescriptor | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderAlert = async (): Promise<void> => {
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <KnownBrowserIssuesAlert />
      </ChakraProvider>
    );
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  });
};

beforeEach(() => {
  detectKnownBrowserIssuesMock.mockReset();
  detectKnownBrowserIssuesMock.mockResolvedValue([]);
  originalClipboardDescriptor = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;

  if (originalClipboardDescriptor) {
    Object.defineProperty(navigator, 'clipboard', originalClipboardDescriptor);
  } else {
    Reflect.deleteProperty(navigator, 'clipboard');
  }
});

describe('known browser issues alert', () => {
  it('stays hidden in an unmodified Chromium environment without attaching its probe canvas', async () => {
    const actualModule = await vi.importActual<typeof knownBrowserIssuesModule>('@platform/browser/knownBrowserIssues');
    detectKnownBrowserIssuesMock.mockImplementationOnce(actualModule.detectKnownBrowserIssues);
    const bodyChildCount = document.body.childElementCount;

    await renderAlert();
    await vi.waitFor(() => expect(detectKnownBrowserIssuesMock).toHaveBeenCalledOnce());

    expect(host?.querySelector('[role="alert"]')).toBeNull();
    expect(document.body.childElementCount).toBe(bodyChildCount);
  });

  it('renders the detected issue with accessible semantics and both workaround paths', async () => {
    detectKnownBrowserIssuesMock.mockResolvedValue([KNOWN_BROWSER_ISSUES[0]]);

    await renderAlert();
    await vi.waitFor(() => expect(host?.textContent).toContain('Canvas protection may corrupt painted strokes'));

    const alert = host?.querySelector('[role="alert"]');
    const copyButton = host?.querySelector<HTMLButtonElement>('button[aria-label="Copy browser setting"]');

    expect(alert).not.toBeNull();
    expect(alert?.textContent).toContain('later strokes noisy or pixelated');
    expect(alert?.textContent).toContain('disable “[Helium Noise] Canvas pixel noising”');
    expect(alert?.textContent).toContain('canvas-protection extension');
    expect(alert?.textContent).toContain('helium://flags/#helium-noise-canvas');
    expect(copyButton?.title).toBe('Copy browser setting');
    expect(host?.querySelectorAll('button')).toHaveLength(1);
  });

  it('copies the Helium setting address', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });
    detectKnownBrowserIssuesMock.mockResolvedValue([KNOWN_BROWSER_ISSUES[0]]);

    await renderAlert();
    const copyButton = await vi.waitFor(() => {
      const button = host?.querySelector<HTMLButtonElement>('button[aria-label="Copy browser setting"]');
      expect(button).not.toBeNull();
      return button!;
    });
    await act(() => userEvent.click(copyButton));

    expect(writeText).toHaveBeenCalledWith('helium://flags/#helium-noise-canvas');
  });

  it('ignores detection results that resolve after unmount', async () => {
    let resolveDetection: ((issues: readonly (typeof KNOWN_BROWSER_ISSUES)[number][]) => void) | undefined;
    detectKnownBrowserIssuesMock.mockReturnValueOnce(
      new Promise<readonly (typeof KNOWN_BROWSER_ISSUES)[number][]>((resolve) => {
        resolveDetection = resolve;
      })
    );

    await renderAlert();
    await act(() => root?.unmount());
    root = null;
    await act(async () => {
      resolveDetection?.([KNOWN_BROWSER_ISSUES[0]]);
      await Promise.resolve();
    });

    expect(host?.childElementCount).toBe(0);
  });
});
