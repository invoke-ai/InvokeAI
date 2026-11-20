import { describe, expect, it } from 'vitest';

import {
  type BrowserFamily,
  type BrowserIssueDetectionEnvironment,
  detectKnownBrowserIssues,
  identifyBrowserFamily,
  KNOWN_BROWSER_ISSUES,
} from './knownBrowserIssues';

interface TestEnvironmentOptions {
  browserFamily?: BrowserFamily;
  hasWebLocks?: boolean;
  isContextAvailable?: boolean;
  mutateReadback?: (pixels: Uint8ClampedArray) => void;
  readbackError?: Error;
}

const createTestEnvironment = ({
  browserFamily = 'chromium',
  hasWebLocks = true,
  isContextAvailable = true,
  mutateReadback,
  readbackError,
}: TestEnvironmentOptions = {}): BrowserIssueDetectionEnvironment => {
  let writtenPixels = new Uint8ClampedArray();
  const context = {
    createImageData: (width: number, height: number) =>
      ({
        data: new Uint8ClampedArray(width * height * 4),
        height,
        width,
      }) as ImageData,
    getImageData: () => {
      if (readbackError) {
        throw readbackError;
      }

      const pixels = new Uint8ClampedArray(writtenPixels);
      mutateReadback?.(pixels);

      return { data: pixels } as ImageData;
    },
    putImageData: (imageData: ImageData) => {
      writtenPixels = new Uint8ClampedArray(imageData.data);
    },
  } as unknown as CanvasRenderingContext2D;
  const canvas = {
    getContext: () => (isContextAvailable ? context : null),
    height: 0,
    width: 0,
  } as unknown as HTMLCanvasElement;

  return {
    browserFamily,
    createCanvas: () => canvas,
    hasWebLocks: () => hasWebLocks,
  };
};

const noisyReadback = (pixels: Uint8ClampedArray): void => {
  pixels[0] = pixels[0] === 255 ? 254 : pixels[0] + 1;
};

const getDetectedIssueIds = (environment: BrowserIssueDetectionEnvironment): readonly string[] =>
  detectKnownBrowserIssues(environment).map((issue) => issue.id);

const getWorkaroundIds = (browserFamily: BrowserFamily): readonly string[] =>
  detectKnownBrowserIssues(createTestEnvironment({ browserFamily, mutateReadback: noisyReadback })).flatMap((issue) =>
    issue.workarounds.map((workaround) => workaround.id)
  );

const CHROME_USER_AGENT =
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36';
const FIREFOX_USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64; rv:142.0) Gecko/20100101 Firefox/142.0';
const SAFARI_USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15';

describe('known browser issues', () => {
  it.each<[string, TestEnvironmentOptions, readonly string[]]>([
    ['accepts an exact canvas pixel readback', {}, []],
    ['detects a modified RGB byte', { mutateReadback: noisyReadback }, ['canvas-readback-integrity']],
    [
      'detects a modified alpha byte',
      {
        mutateReadback: (pixels) => {
          pixels[3] = 254;
        },
      },
      ['canvas-readback-integrity'],
    ],
    ['detects an unavailable 2D context', { isContextAvailable: false }, ['canvas-readback-integrity']],
    [
      'detects a blocked canvas readback',
      { readbackError: new Error('Canvas readback blocked') },
      ['canvas-readback-integrity'],
    ],
  ])('%s', (_label, options, expected) => {
    expect(getDetectedIssueIds(createTestEnvironment(options))).toEqual(expected);
  });

  it('returns registry metadata without the detector', () => {
    const [detectedIssue] = detectKnownBrowserIssues(createTestEnvironment({ mutateReadback: noisyReadback }));
    const { detect: _detect, ...registryIssue } = KNOWN_BROWSER_ISSUES[0];

    expect(detectedIssue).not.toHaveProperty('detect');
    expect(detectedIssue).toEqual({
      ...registryIssue,
      workarounds: registryIssue.workarounds.filter(
        (workaround) => workaround.id !== 'brave' && workaround.id !== 'firefox'
      ),
    });
  });

  it.each<[string, BrowserFamily, readonly string[]]>([
    ['shows Brave users only the Shields instruction plus the generic fallback', 'brave', ['brave', 'other-browser']],
    [
      'shows Firefox users the canvas permission instruction instead of the Helium flag',
      'firefox',
      ['firefox', 'other-browser'],
    ],
    [
      'keeps the Helium flag for Chromium builds that cannot be told apart from Helium',
      'chromium',
      ['helium', 'other-browser'],
    ],
    ['shows only the generic instruction elsewhere', 'other', ['other-browser']],
  ])('%s', (_label, browserFamily, expected) => {
    expect(getWorkaroundIds(browserFamily)).toEqual(expected);
  });

  it('detects the plain-HTTP limits whenever Web Locks are unavailable, and only then', () => {
    // A secure context can lack Web Locks too (an old browser, a disabled API): the notice follows the capability.
    expect(getDetectedIssueIds(createTestEnvironment({ hasWebLocks: false }))).toEqual(['plain-http-limits']);
    expect(getDetectedIssueIds(createTestEnvironment({ hasWebLocks: true }))).toEqual([]);
    expect(
      detectKnownBrowserIssues(createTestEnvironment({ hasWebLocks: false })).map(({ dismissible, id }) => ({
        dismissible,
        id,
      }))
    ).toEqual([{ dismissible: true, id: 'plain-http-limits' }]);
  });
});

describe('identifyBrowserFamily', () => {
  it('recognizes Brave by its navigator API ahead of its Chrome user agent', () => {
    expect(
      identifyBrowserFamily({ brave: { isBrave: () => Promise.resolve(true) }, userAgent: CHROME_USER_AGENT })
    ).toBe('brave');
  });

  it('does not treat a non-callable brave property as Brave', () => {
    expect(identifyBrowserFamily({ brave: {}, userAgent: CHROME_USER_AGENT })).toBe('chromium');
  });

  it('recognizes Firefox-based browsers', () => {
    expect(identifyBrowserFamily({ userAgent: FIREFOX_USER_AGENT })).toBe('firefox');
  });

  it('falls back to other for WebKit browsers', () => {
    expect(identifyBrowserFamily({ userAgent: SAFARI_USER_AGENT })).toBe('other');
  });
});
