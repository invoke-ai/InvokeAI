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
  isContextAvailable?: boolean;
  mutateReadback?: (pixels: Uint8ClampedArray) => void;
  readbackError?: Error;
}

const createTestEnvironment = ({
  browserFamily = 'chromium',
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
  it('accepts an exact canvas pixel readback', () => {
    expect(getDetectedIssueIds(createTestEnvironment())).toEqual([]);
  });

  it('detects a modified RGB byte', () => {
    expect(getDetectedIssueIds(createTestEnvironment({ mutateReadback: noisyReadback }))).toEqual([
      'canvas-readback-integrity',
    ]);
  });

  it('detects a modified alpha byte', () => {
    const environment = createTestEnvironment({
      mutateReadback: (pixels) => {
        pixels[3] = 254;
      },
    });

    expect(getDetectedIssueIds(environment)).toEqual(['canvas-readback-integrity']);
  });

  it('detects an unavailable 2D context', () => {
    expect(getDetectedIssueIds(createTestEnvironment({ isContextAvailable: false }))).toEqual([
      'canvas-readback-integrity',
    ]);
  });

  it('detects a blocked canvas readback', () => {
    expect(getDetectedIssueIds(createTestEnvironment({ readbackError: new Error('Canvas readback blocked') }))).toEqual(
      ['canvas-readback-integrity']
    );
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

  it('shows Brave users only the Shields instruction plus the generic fallback', () => {
    expect(getWorkaroundIds('brave')).toEqual(['brave', 'other-browser']);
  });

  it('shows Firefox users the canvas permission instruction instead of the Helium flag', () => {
    expect(getWorkaroundIds('firefox')).toEqual(['firefox', 'other-browser']);
  });

  it('keeps the Helium flag for Chromium builds that cannot be told apart from Helium', () => {
    expect(getWorkaroundIds('chromium')).toEqual(['helium', 'other-browser']);
  });

  it('shows only the generic instruction elsewhere', () => {
    expect(getWorkaroundIds('other')).toEqual(['other-browser']);
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
