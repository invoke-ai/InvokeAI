// @vitest-environment happy-dom
/**
 * Mounted-DOM coverage for the preview components' reveal wiring. The controller's sequencing has
 * its own unit tests; what only a real mount can verify is the wiring the components rely on —
 * effect ordering, cleanup, StrictMode double-invocation, the image <-> video component swap over
 * the shared ref, and media readiness driven by a real <video> element's event.
 */
import { createAutoSwitchedSelectionMarker } from 'features/gallery/store/autoSwitchedImages';
import { atom } from 'nanostores';
import { act, StrictMode } from 'react';
import type { Root } from 'react-dom/client';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { usePaintedItemName, useSelectedItemReveal } from './useSelectedItemReveal';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const DURATION_MS = 2000;
const MEDIA_GRACE_MS = 1000;

/** The pieces the viewer context shares between the two preview components. */
const createSharedContext = () => ({
  lastRenderedItemNameRef: { current: null as string | null },
  $isTemporarilyShowingSelectedImage: atom(false),
  marker: createAutoSwitchedSelectionMarker(),
});

type Shared = ReturnType<typeof createSharedContext>;

const RevealProbe = ({
  shared,
  itemName,
  isMediaReady = true,
  hasProgressImage = true,
  isProgressImageResolving = false,
}: {
  shared: Shared;
  itemName: string | null;
  isMediaReady?: boolean;
  hasProgressImage?: boolean;
  isProgressImageResolving?: boolean;
}) => {
  useSelectedItemReveal({
    ...shared,
    durationMs: DURATION_MS,
    mediaGraceMs: MEDIA_GRACE_MS,
    renderedItemName: itemName,
    isMediaReady,
    selectedItemName: itemName,
    shouldShowProgressInViewer: true,
    hasProgressImage,
    isProgressImageResolving,
  });
  return null;
};

/** CurrentVideoPreview's readiness wiring: a real <video> element feeding usePaintedItemName. */
const VideoProbe = ({ shared, videoName }: { shared: Shared; videoName: string | null }) => {
  const { isMediaReady, onPainted } = usePaintedItemName(videoName);
  useSelectedItemReveal({
    ...shared,
    durationMs: DURATION_MS,
    mediaGraceMs: MEDIA_GRACE_MS,
    renderedItemName: videoName,
    isMediaReady,
    selectedItemName: videoName,
    shouldShowProgressInViewer: true,
    hasProgressImage: true,
    isProgressImageResolving: false,
  });
  return <video data-testid="probe-video" key={videoName ?? undefined} onLoadedData={onPainted} />;
};

describe('useSelectedItemReveal (mounted)', () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    vi.useFakeTimers();
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    vi.useRealTimers();
  });

  it('reveals a selection change and lowers when the duration elapses', () => {
    const shared = createSharedContext();
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="a.png" />);
    });
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="b.png" />);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get()).toBe(true);

    act(() => {
      vi.advanceTimersByTime(DURATION_MS);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get()).toBe(false);
  });

  it('lowers the flag on unmount and leaves no timer to re-raise it', () => {
    const shared = createSharedContext();
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="a.png" />);
    });
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="b.png" />);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get()).toBe(true);

    act(() => {
      root.unmount();
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get(), 'a reveal must not outlive its component').toBe(false);

    act(() => {
      vi.advanceTimersByTime(DURATION_MS * 2);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get()).toBe(false);
    root = createRoot(container); // afterEach unmounts a root; give it a live one
  });

  it('survives StrictMode double-invoked effects', () => {
    const shared = createSharedContext();
    act(() => {
      root.render(
        <StrictMode>
          <RevealProbe shared={shared} itemName="a.png" />
        </StrictMode>
      );
    });
    act(() => {
      root.render(
        <StrictMode>
          <RevealProbe shared={shared} itemName="b.png" />
        </StrictMode>
      );
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get(), 'the reveal survives the doubled effects').toBe(true);
  });

  it('carries the reveal across an image -> video component swap', () => {
    // The two preview components are mutually exclusive; the shared ref is what makes a click that
    // switches media type read as a selection change. Only a real unmount/mount can test that the
    // outgoing component's cleanup does not destroy the incoming one's reveal.
    const shared = createSharedContext();
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="a.png" />);
    });
    act(() => {
      root.render(<VideoProbe shared={shared} videoName="b.mp4" />);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get(), 'held for the unpainted video').toBe(false);

    const video = container.querySelector('video');
    expect(video).not.toBeNull();
    act(() => {
      video?.dispatchEvent(new Event('loadeddata'));
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get(), 'revealed once the frame is there').toBe(true);
  });

  it('does not reveal an auto-switched selection', () => {
    const shared = createSharedContext();
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="a.png" />);
    });
    shared.marker.record('b.png');
    shared.marker.settle('b.png');
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="b.png" />);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get()).toBe(false);
  });

  it('reveals an unpainted video after the media grace even if loadeddata never fires', () => {
    const shared = createSharedContext();
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="a.png" />);
    });
    act(() => {
      root.render(<VideoProbe shared={shared} videoName="never-loads.mp4" />);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get()).toBe(false);
    act(() => {
      vi.advanceTimersByTime(MEDIA_GRACE_MS);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get(), 'the click still lands').toBe(true);
  });

  it('resets readiness when the video element is swapped for another video', () => {
    // A stale "painted" from the previous element must not lift the overlay onto the new, black
    // one — the readiness is compared by name, so the swap cannot inherit it.
    const shared = createSharedContext();
    act(() => {
      root.render(<VideoProbe shared={shared} videoName="a.mp4" />);
    });
    act(() => {
      container.querySelector('video')?.dispatchEvent(new Event('loadeddata'));
    });
    act(() => {
      root.render(<VideoProbe shared={shared} videoName="b.mp4" />);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get(), 'held: the new element has not painted').toBe(false);
    act(() => {
      container.querySelector('video')?.dispatchEvent(new Event('loadeddata'));
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get()).toBe(true);
  });

  it("does not let the outgoing component's timer cut the incoming reveal short", () => {
    // Image A is mid-reveal when the user clicks video B. A's controller unmounts with ~1.5s left
    // on its timer; if the cleanup does not cancel it, that timer fires into the shared flag and
    // ends B's two seconds early.
    const shared = createSharedContext();
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="a.png" />);
    });
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="b.png" />);
    });
    act(() => {
      vi.advanceTimersByTime(500);
    });
    act(() => {
      root.render(<RevealProbe shared={shared} itemName="c.png" key="other-component" />);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get()).toBe(true);

    // Past where the outgoing component's timer would have fired, short of the new reveal's end.
    act(() => {
      vi.advanceTimersByTime(DURATION_MS - 400);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get(), 'the new reveal runs its full duration').toBe(true);

    act(() => {
      vi.advanceTimersByTime(400);
    });
    expect(shared.$isTemporarilyShowingSelectedImage.get()).toBe(false);
  });
});
