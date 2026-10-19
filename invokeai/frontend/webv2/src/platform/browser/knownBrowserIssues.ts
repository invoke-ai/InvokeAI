export type KnownBrowserIssueId = 'canvas-readback-integrity';

export type KnownBrowserIssueSeverity = 'warning' | 'error';

export type BrowserFamily = 'brave' | 'chromium' | 'firefox' | 'other';

export interface KnownBrowserIssueWorkaround {
  id: string;
  instructionKey: string;
  copyableValue?: string;
  /** Families this instruction applies to; omitted means every browser. */
  browsers?: readonly BrowserFamily[];
}

export interface BrowserIssueDetectionEnvironment {
  browserFamily: BrowserFamily;
  createCanvas: () => HTMLCanvasElement;
}

export interface DetectedBrowserIssue {
  id: KnownBrowserIssueId;
  severity: KnownBrowserIssueSeverity;
  titleKey: string;
  descriptionKey: string;
  workarounds: readonly KnownBrowserIssueWorkaround[];
}

export interface KnownBrowserIssue extends DetectedBrowserIssue {
  detect: (environment: BrowserIssueDetectionEnvironment) => boolean;
}

interface BrowserIdentity {
  brave?: { isBrave?: unknown };
  userAgent: string;
}

export const identifyBrowserFamily = ({ brave, userAgent }: BrowserIdentity): BrowserFamily => {
  if (typeof brave?.isBrave === 'function') {
    return 'brave';
  }

  if (userAgent.includes('Firefox/')) {
    return 'firefox';
  }

  if (userAgent.includes('Chrome/')) {
    return 'chromium';
  }

  return 'other';
};

const CANVAS_PROBE_SIZE = 32;

const createDefaultDetectionEnvironment = (): BrowserIssueDetectionEnvironment => ({
  browserFamily: identifyBrowserFamily(navigator as Navigator & BrowserIdentity),
  createCanvas: () => document.createElement('canvas'),
});

const createExpectedCanvasPixels = (): Uint8ClampedArray => {
  const pixels = new Uint8ClampedArray(CANVAS_PROBE_SIZE * CANVAS_PROBE_SIZE * 4);

  for (let pixelIndex = 0; pixelIndex < CANVAS_PROBE_SIZE * CANVAS_PROBE_SIZE; pixelIndex++) {
    const byteIndex = pixelIndex * 4;

    pixels[byteIndex] = 32 + ((pixelIndex * 17) % 192);
    pixels[byteIndex + 1] = 32 + ((pixelIndex * 31) % 192);
    pixels[byteIndex + 2] = 32 + ((pixelIndex * 47) % 192);
    pixels[byteIndex + 3] = 255;
  }

  return pixels;
};

const hasUnsafeCanvasReadback = (environment: BrowserIssueDetectionEnvironment): boolean => {
  try {
    const canvas = environment.createCanvas();
    canvas.width = CANVAS_PROBE_SIZE;
    canvas.height = CANVAS_PROBE_SIZE;

    const context = canvas.getContext('2d');

    if (!context) {
      return true;
    }

    const expectedPixels = createExpectedCanvasPixels();
    const imageData = context.createImageData(CANVAS_PROBE_SIZE, CANVAS_PROBE_SIZE);
    imageData.data.set(expectedPixels);
    context.putImageData(imageData, 0, 0);

    const actualPixels = context.getImageData(0, 0, CANVAS_PROBE_SIZE, CANVAS_PROBE_SIZE).data;

    if (actualPixels.length !== expectedPixels.length) {
      return true;
    }

    return actualPixels.some((value, index) => value !== expectedPixels[index]);
  } catch {
    return true;
  }
};

export const KNOWN_BROWSER_ISSUES = [
  {
    descriptionKey: 'launchpad.browserIssues.canvasReadbackIntegrity.description',
    detect: hasUnsafeCanvasReadback,
    id: 'canvas-readback-integrity',
    severity: 'warning',
    titleKey: 'launchpad.browserIssues.canvasReadbackIntegrity.title',
    workarounds: [
      {
        browsers: ['brave'],
        id: 'brave',
        instructionKey: 'launchpad.browserIssues.canvasReadbackIntegrity.braveWorkaround',
      },
      {
        browsers: ['firefox'],
        id: 'firefox',
        instructionKey: 'launchpad.browserIssues.canvasReadbackIntegrity.firefoxWorkaround',
      },
      {
        browsers: ['chromium'],
        copyableValue: 'helium://flags/#helium-noise-canvas',
        id: 'helium',
        instructionKey: 'launchpad.browserIssues.canvasReadbackIntegrity.heliumWorkaround',
      },
      {
        id: 'other-browser',
        instructionKey: 'launchpad.browserIssues.canvasReadbackIntegrity.genericWorkaround',
      },
    ],
  },
] as const satisfies readonly KnownBrowserIssue[];

const REGISTERED_ISSUES: readonly KnownBrowserIssue[] = KNOWN_BROWSER_ISSUES;

export const detectKnownBrowserIssues = (
  environment: BrowserIssueDetectionEnvironment = createDefaultDetectionEnvironment()
): readonly DetectedBrowserIssue[] =>
  REGISTERED_ISSUES.flatMap(({ detect, ...issue }): DetectedBrowserIssue[] =>
    detect(environment)
      ? [
          {
            ...issue,
            workarounds: issue.workarounds.filter(
              (workaround) => !workaround.browsers || workaround.browsers.includes(environment.browserFamily)
            ),
          },
        ]
      : []
  );
