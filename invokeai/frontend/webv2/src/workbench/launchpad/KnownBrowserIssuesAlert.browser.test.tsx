import type * as knownBrowserIssuesModule from '@platform/browser/knownBrowserIssues';

import { ChakraProvider } from '@chakra-ui/react';
import { type DetectedBrowserIssue, KNOWN_BROWSER_ISSUES } from '@platform/browser/knownBrowserIssues';
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
  'launchpad.browserIssues.canvasReadbackIntegrity.braveWorkaround':
    'Click the Brave Shields icon in the address bar and turn Shields off for this site, then reload Invoke.',
  'launchpad.browserIssues.canvasReadbackIntegrity.description':
    'Your browser is modifying or blocking canvas pixel data. This can make later strokes noisy or pixelated.',
  'launchpad.browserIssues.canvasReadbackIntegrity.genericWorkaround':
    'If you use another privacy browser or canvas-protection extension, allow unmodified canvas access for this site, then reload Invoke.',
  'launchpad.browserIssues.canvasReadbackIntegrity.heliumWorkaround':
    'In Helium, open the address below, disable “[Helium Noise] Canvas pixel noising”, then relaunch Helium.',
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

const [CANVAS_ISSUE] = KNOWN_BROWSER_ISSUES;

const createDetectedIssue = (...workaroundIds: readonly string[]): DetectedBrowserIssue => {
  const { detect: _detect, ...issue } = CANVAS_ISSUE;

  return {
    ...issue,
    workarounds: issue.workarounds.filter((workaround) => workaroundIds.includes(workaround.id)),
  };
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let originalClipboardDescriptor: PropertyDescriptor | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderAlert = async (): Promise<void> => {
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <KnownBrowserIssuesAlert />
      </ChakraProvider>
    );
  });
};

beforeEach(() => {
  detectKnownBrowserIssuesMock.mockReset();
  detectKnownBrowserIssuesMock.mockReturnValue([]);
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

    expect(detectKnownBrowserIssuesMock).toHaveBeenCalledOnce();
    expect(host?.querySelector('[role="alert"]')).toBeNull();
    expect(document.body.childElementCount).toBe(bodyChildCount);
  });

  it('renders the detected issue with accessible semantics and the Helium workaround', async () => {
    detectKnownBrowserIssuesMock.mockReturnValue([createDetectedIssue('helium', 'other-browser')]);

    await renderAlert();

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

  it('shows Brave users the Shields instruction without a copyable flag', async () => {
    detectKnownBrowserIssuesMock.mockReturnValue([createDetectedIssue('brave', 'other-browser')]);

    await renderAlert();

    const alert = host?.querySelector('[role="alert"]');

    expect(alert?.textContent).toContain('turn Shields off for this site');
    expect(alert?.textContent).not.toContain('Helium');
    expect(host?.querySelectorAll('button')).toHaveLength(0);
  });

  it('copies the Helium setting address', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });
    detectKnownBrowserIssuesMock.mockReturnValue([createDetectedIssue('helium')]);

    await renderAlert();
    const copyButton = host?.querySelector<HTMLButtonElement>('button[aria-label="Copy browser setting"]');

    expect(copyButton).not.toBeNull();
    await act(() => userEvent.click(copyButton!));

    expect(writeText).toHaveBeenCalledWith('helium://flags/#helium-noise-canvas');
  });
});
