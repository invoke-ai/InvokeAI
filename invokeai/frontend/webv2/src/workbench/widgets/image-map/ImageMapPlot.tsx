import type { ImageMapPoint } from '@workbench/image-map/api';
import type { AxisRanges } from '@workbench/image-map/imageMapViewport';
import type { PlotlyHTMLElement } from 'plotly.js';

import { Box } from '@chakra-ui/react';
import {
  getPersistedSelectedGalleryItemKeys,
  getSelectedGalleryImageFromValues,
  parseGalleryItemKey,
} from '@features/gallery/contracts';
import { attachWheelZoom } from '@workbench/image-map/attachWheelZoom';
import { collectClusterSelection } from '@workbench/image-map/clusterSelection';
import { imageMapStore } from '@workbench/image-map/imageMapStore';
import {
  buildAllPointsTrace,
  buildClusterAnnotations,
  buildCurrentImageTrace,
  buildHighlightedPointsTrace,
  buildMapLayout,
  CURRENT_IMAGE_TRACE,
  HIGHLIGHTED_POINTS_TRACE,
} from '@workbench/image-map/imageMapTraces';
import {
  computePercentileRanges,
  expandRangesToInclude,
  fitRangesToAspect,
  rangesToKeepMarkerInView,
} from '@workbench/image-map/imageMapViewport';
import { getThumbnailUrl } from '@workbench/image-map/thumbnailCache';
import { shallowEqual, useWidgetValuesSelector } from '@workbench/WorkbenchContext';
import Plotly from 'plotly.js-gl2d-dist-min';
import { useEffect, useMemo, useRef, useState } from 'react';

import { useMapSelection } from './useSelectMapImage';

/** Suppress the synthetic click plotly fires when a pinch gesture ends. */
const PINCH_CLICK_SUPPRESS_MS = 500;

/** How long a map click may suppress the recenter its own selection causes. */
const MAP_CLICK_SUPPRESS_MS = 5000;

/** Dwell before a hover thumbnail appears (PhotoMapAI's delay). */
const HOVER_DELAY_MS = 150;

/** These plotly calls can reject on a plot whose WebGL init failed; the map
 * already shows the store's error state, so the rejection itself is noise. */
const swallow = (promise: Promise<unknown>): void => {
  promise.catch(() => {});
};

/** Matches the preview's `maxW`/`maxH` (Chakra 40 = 10rem); used to keep it on-screen. */
const HOVER_PREVIEW_MAX_PX = 160;
const HOVER_PREVIEW_OFFSET_PX = 14;

interface HoverPreview {
  imageName: string;
  url: string;
  clientX: number;
  clientY: number;
}

interface PlotElement extends PlotlyHTMLElement {
  _fullLayout?: {
    xaxis?: { range?: [number, number] };
    yaxis?: { range?: [number, number] };
  };
}

const readRanges = (plot: PlotElement): AxisRanges | null => {
  const x = plot._fullLayout?.xaxis?.range;
  const y = plot._fullLayout?.yaxis?.range;

  return x && y ? { x: [x[0], x[1]], y: [y[0], y[1]] } : null;
};

/**
 * The whole-map view for the first properly-sized render: the percentile box
 * expanded to include the current-image marker (so the auto-recenter has no
 * reason to immediately shift it), then aspect-corrected to the container.
 * The axes are constrained to equal unit scale, and letting plotly resolve an
 * over-constrained range pair itself can crop one axis — a first render in a
 * still-unmeasured container ends up zoomed into a sliver of the map, which
 * the view-preservation on later renders would then keep forever.
 */
const computeInitialFit = (
  points: ImageMapPoint[],
  selectedImageName: string | null,
  width: number,
  height: number
): AxisRanges | null => {
  if (points.length === 0 || width <= 0 || height <= 0) {
    return null;
  }

  let box = computePercentileRanges(points);

  if (!box) {
    return null;
  }

  const selected = selectedImageName
    ? points.find((candidate) => candidate.imageName === selectedImageName)
    : undefined;

  if (selected) {
    box = expandRangesToInclude(box, selected);
  }

  return fitRangesToAspect(box, width / height);
};

const findTraceIndex = (plot: PlotElement, name: string): number =>
  (plot.data ?? []).findIndex((trace) => (trace as { name?: string }).name === name);

/**
 * Imperative plotly host. All plotly calls happen in effects against a ref
 * div — plotly manages its own DOM and must never render through JSX. This
 * module is lazy-loaded so the plotly bundle stays out of the app's critical
 * path.
 */
const ImageMapPlot = ({
  clickSelectsCluster = false,
  showClusterLabels = true,
}: {
  clickSelectsCluster?: boolean;
  showClusterLabels?: boolean;
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const points = imageMapStore.useSelector((snapshot) => snapshot.data?.points ?? null);
  const clusterLabels = imageMapStore.useSelector((snapshot) => snapshot.clusterLabels);
  const selectedImageName = useWidgetValuesSelector(
    'gallery',
    (values) => getSelectedGalleryImageFromValues(values)?.imageName ?? null
  );
  // The persisted selection stores kind-tagged item keys; map points carry
  // bare image names, so parse the keys back down (videos never plot).
  const selectedImageNames = useWidgetValuesSelector(
    'gallery',
    (values) =>
      getPersistedSelectedGalleryItemKeys(values)
        .map(parseGalleryItemKey)
        .filter((ref) => ref.kind === 'image')
        .map((ref) => ref.name),
    shallowEqual
  );
  const selectedNames = useMemo(() => new Set(selectedImageNames), [selectedImageNames]);
  const { selectCluster, selectImage } = useMapSelection();
  // Bumped after every scene rebuild so the overlay effects (marker,
  // highlight) re-apply onto the fresh, empty overlay traces.
  const [plotRevision, setPlotRevision] = useState(0);
  const lastPinchAtRef = useRef(0);
  const lastMapSelectionRef = useRef<{ name: string; at: number } | null>(null);
  // The initial whole-map fit must happen exactly once per mount, at the
  // first render where the container has real dimensions; these refs let the
  // scene effect and the resize observer coordinate without re-running.
  const initialFitDoneRef = useRef(false);
  const pointsRef = useRef(points);
  const selectedImageNameRef = useRef(selectedImageName);
  const clusterModeRef = useRef(clickSelectsCluster);

  // Declared before the effects below so the refs are fresh when they run.
  useEffect(() => {
    pointsRef.current = points;
    selectedImageNameRef.current = selectedImageName;
    clusterModeRef.current = clickSelectsCluster;
  }, [clickSelectsCluster, points, selectedImageName]);
  const [pendingHoverPreview, setHoverPreview] = useState<HoverPreview | null>(null);
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Monotonic hover session: a resolution from a previous hover (even of the
  // same point) must neither show early nor at stale coordinates.
  const hoverSessionRef = useRef(0);

  const clearHover = () => {
    hoverSessionRef.current += 1;
    if (hoverTimerRef.current !== null) {
      clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    setHoverPreview(null);
  };

  useEffect(() => {
    const container = containerRef.current;

    if (!container || points === null) {
      return;
    }

    // Overlay traces (highlight, marker) start empty; the overlay effects
    // below restyle them, so a selection change never rebuilds the scene.
    const traces = [
      buildAllPointsTrace(points),
      buildHighlightedPointsTrace(points, new Set()),
      buildCurrentImageTrace(),
    ];
    let disposed = false;

    // Feed the CURRENT view back into react so data refreshes never reset the
    // user's pan/zoom (uirevision alone does not preserve ranges set through
    // the public relayout API). The first properly-sized render instead gets
    // an aspect-corrected whole-map fit — never a preserved view, which
    // could be a cropped artifact of a zero-size initial layout.
    let initialRanges = readRanges(container as unknown as PlotElement) ?? computePercentileRanges(points);

    if (!initialFitDoneRef.current) {
      const fitted = computeInitialFit(
        points,
        selectedImageNameRef.current,
        container.offsetWidth,
        container.offsetHeight
      );

      if (fitted) {
        initialRanges = fitted;
        initialFitDoneRef.current = true;
      }
    }

    const layout = buildMapLayout(
      initialRanges,
      buildClusterAnnotations(points, showClusterLabels ? clusterLabels : null)
    );

    void Plotly.react(container, traces as Plotly.Data[], layout, {
      displayModeBar: false,
      // Custom wheel/pinch zoom below; plotly's own scrollZoom has
      // long-standing Safari issues.
      scrollZoom: false,
    })
      .then((plot: PlotlyHTMLElement) => {
        if (disposed) {
          return;
        }

        plot.removeAllListeners?.('plotly_click');
        plot.on('plotly_click', (event) => {
          if (Date.now() - lastPinchAtRef.current < PINCH_CLICK_SUPPRESS_MS) {
            return;
          }

          const imageName = event.points?.[0]?.customdata;

          if (typeof imageName !== 'string') {
            return;
          }

          // A selection made by clicking the map must not recenter the map
          // under the user's cursor; the marker effect checks this. The
          // stamp expires so a stale entry (failed hydrate, re-click of the
          // current point) cannot suppress a legitimate future recenter.
          lastMapSelectionRef.current = { at: Date.now(), name: imageName };

          const clusterNames = clusterModeRef.current ? collectClusterSelection(points, imageName) : null;

          if (clusterNames) {
            selectCluster(imageName, clusterNames);
          } else {
            // Also the cluster-mode fallback for noise points (cluster -1).
            selectImage(imageName);
          }
        });
        plot.removeAllListeners?.('plotly_hover');
        plot.on('plotly_hover', (event) => {
          const imageName = event.points?.[0]?.customdata;
          const mouse = (event as { event?: MouseEvent }).event;

          if (typeof imageName !== 'string' || !mouse) {
            return;
          }

          clearHover();
          const session = hoverSessionRef.current;
          const { clientX, clientY } = mouse;
          hoverTimerRef.current = setTimeout(() => {
            void getThumbnailUrl(imageName).then((url) => {
              // Deliberately not gated on `disposed`: that belongs to this
              // effect, which re-runs on every socket-driven refresh, so a
              // refresh landing mid-dwell would drop the thumbnail — and no
              // new `plotly_hover` fires while the pointer sits still.
              // Unmount is covered by the session bump in the cleanup below.
              if (url && hoverSessionRef.current === session) {
                setHoverPreview({ clientX, clientY, imageName, url });
              }
            });
          }, HOVER_DELAY_MS);
        });
        plot.removeAllListeners?.('plotly_unhover');
        plot.on('plotly_unhover', () => {
          clearHover();
        });
        setPlotRevision((revision) => revision + 1);
      })
      .catch(() => {
        if (disposed) {
          // Symmetry with the .then above: a rejection arriving after unmount
          // must not write a global error on behalf of a dead component.
          return;
        }

        // WebGL context creation can fail (blocked GPU, context exhaustion).
        // Reported as renderError, not the generic error: the data is fine, it
        // is the canvas that is not, so the view has to stop trying to render
        // the plot. Signalling this through `error`/`loadState` alone did
        // nothing, because the view prefers a non-empty point set over any
        // error and would just mount this same failing plot again.
        imageMapStore.patchSnapshot({ renderError: 'The map failed to render (WebGL unavailable).' });
      });

    return () => {
      disposed = true;
    };
  }, [clusterLabels, points, selectCluster, selectImage, showClusterLabels]);

  // Highlight overlay: the gallery's multi-selection, restyled in place.
  useEffect(() => {
    const container = containerRef.current as PlotElement | null;

    if (!container || points === null) {
      return;
    }

    const highlightIndex = findTraceIndex(container, HIGHLIGHTED_POINTS_TRACE);

    if (highlightIndex < 0) {
      return;
    }

    const trace = buildHighlightedPointsTrace(points, selectedNames);
    swallow(
      Plotly.restyle(
        container,
        {
          customdata: [trace.customdata],
          'marker.color': [trace.marker.color as string[]],
          x: [trace.x],
          y: [trace.y],
        },
        [highlightIndex]
      )
    );
  }, [plotRevision, points, selectedNames]);

  // The timer must not outlive the component; the scene effect used to do
  // this, but it reruns on every `points` change.
  useEffect(() => clearHover, []);

  // Ending the session retires the hover for good. Hiding it alone is not
  // enough: an image that leaves the map never fires `plotly_unhover`, so a
  // later refresh restoring it would pop the thumbnail back up at coordinates
  // captured long before, wherever the pointer has since moved.
  // Live gold target on the currently selected gallery image, with a gentle
  // recenter (zoom width preserved) when it drifts near or beyond an edge.
  useEffect(() => {
    const container = containerRef.current as PlotElement | null;

    if (!container || points === null) {
      return;
    }

    const markerIndex = findTraceIndex(container, CURRENT_IMAGE_TRACE);

    if (markerIndex < 0) {
      return;
    }

    const point = selectedImageName ? points.find((candidate) => candidate.imageName === selectedImageName) : undefined;
    const suppression = lastMapSelectionRef.current;
    const isSuppressionFresh = suppression !== null && Date.now() - suppression.at < MAP_CLICK_SUPPRESS_MS;
    const cameFromMapClick = isSuppressionFresh && suppression.name === selectedImageName;

    // Consume a matching entry; an interleaved external selection keeps a
    // pending map click's suppression intact for when it lands. Expired
    // entries go regardless — a click whose selection never arrived (the
    // image was deleted, or the selection resolved to a non-image) would
    // otherwise sit here and swallow the recenter for a later, unrelated
    // gallery pick of the same name. Both run before the `!point` return,
    // which is the path a never-arriving selection actually takes.
    if (suppression !== null && (cameFromMapClick || !isSuppressionFresh)) {
      lastMapSelectionRef.current = null;
    }

    if (!point) {
      swallow(Plotly.restyle(container, { x: [[]], y: [[]] }, [markerIndex]));

      return;
    }

    swallow(Plotly.restyle(container, { x: [[point.x]], y: [[point.y]] }, [markerIndex]));

    if (cameFromMapClick) {
      return;
    }

    const ranges = readRanges(container);
    const recentered = ranges ? rangesToKeepMarkerInView(ranges, point) : null;

    if (recentered) {
      swallow(Plotly.relayout(container, { 'xaxis.range': recentered.x, 'yaxis.range': recentered.y }));
    }
  }, [plotRevision, points, selectedImageName]);

  // Custom zoom handlers + container size tracking, attached once for the
  // plot's lifetime; plotly does not observe its container.
  useEffect(() => {
    const container = containerRef.current as PlotElement | null;

    if (!container) {
      return;
    }

    const detachZoom = attachWheelZoom(container, {
      applyRanges: (ranges) => {
        swallow(Plotly.relayout(container, { 'xaxis.range': ranges.x, 'yaxis.range': ranges.y }));
      },
      onPinch: () => {
        lastPinchAtRef.current = Date.now();
      },
      readRanges: () => readRanges(container),
    });

    const observer = new ResizeObserver(() => {
      if (container.offsetWidth > 0 && container.offsetHeight > 0) {
        // `@types/plotly.js` declares this `void`; plotly resolves a promise
        // once the resize has actually been applied.
        const resized = Plotly.Plots.resize(container) as unknown as Promise<unknown>;

        // Chained, not fired alongside: the resize defers its own autosize
        // relayout, so a fit applied immediately would be solved against the
        // plot's previous dimensions — and with the axes scale-anchored, that
        // re-solve crops one of them.
        swallow(
          resized.then(() => {
            // A plot first built while the container was unmeasured never got
            // its whole-map fit; apply it on the first real layout.
            if (initialFitDoneRef.current) {
              return undefined;
            }

            const fitted = computeInitialFit(
              pointsRef.current ?? [],
              selectedImageNameRef.current,
              container.offsetWidth,
              container.offsetHeight
            );

            if (!fitted) {
              return undefined;
            }

            initialFitDoneRef.current = true;

            return Plotly.relayout(container, { 'xaxis.range': fitted.x, 'yaxis.range': fitted.y });
          })
        );
      }
    });
    observer.observe(container);

    return () => {
      detachZoom();
      observer.disconnect();
      Plotly.purge(container);
      // Purging drops the layout the flag stands for. Leaving it set costs
      // the whole-map fit on any remount of this same component instance —
      // StrictMode's double-mount in development, for one.
      initialFitDoneRef.current = false;
    };
  }, []);

  // A live refresh can drop the hovered image from the map, which makes the
  // preview stale. Derived from whether the image is still on the map, not
  // from `points` changing: that changes on every socket-driven refresh, and
  // clearing on it would silently cancel live hovers — they would not come
  // back either, since `Plotly.react` resets hover state and no new
  // `plotly_hover` fires until the pointer moves.
  const hoverPreview =
    pendingHoverPreview && points && !points.some((point) => point.imageName === pendingHoverPreview.imageName)
      ? null
      : pendingHoverPreview;

  // The widget is usually docked in a side rail, so a point near the right or
  // bottom edge would otherwise put a fixed-position preview off-screen, where
  // nothing can scroll it into view. Flip to the other side of the cursor when
  // it does not fit.
  const previewPosition = hoverPreview
    ? {
        left:
          hoverPreview.clientX + HOVER_PREVIEW_OFFSET_PX + HOVER_PREVIEW_MAX_PX > window.innerWidth
            ? Math.max(0, hoverPreview.clientX - HOVER_PREVIEW_OFFSET_PX - HOVER_PREVIEW_MAX_PX)
            : hoverPreview.clientX + HOVER_PREVIEW_OFFSET_PX,
        top:
          hoverPreview.clientY + HOVER_PREVIEW_OFFSET_PX + HOVER_PREVIEW_MAX_PX > window.innerHeight
            ? Math.max(0, hoverPreview.clientY - HOVER_PREVIEW_OFFSET_PX - HOVER_PREVIEW_MAX_PX)
            : hoverPreview.clientY + HOVER_PREVIEW_OFFSET_PX,
      }
    : { left: 0, top: 0 };

  return (
    <Box h="full" minH="0" position="relative" w="full">
      <Box ref={containerRef} h="full" w="full" />
      {hoverPreview ? (
        <Box
          borderColor="border.emphasized"
          borderWidth="1px"
          left={`${previewPosition.left}px`}
          maxH="40"
          maxW="40"
          overflow="hidden"
          pointerEvents="none"
          position="fixed"
          rounded="md"
          shadow="lg"
          top={`${previewPosition.top}px`}
          zIndex="tooltip"
        >
          <img
            alt={hoverPreview.imageName}
            src={hoverPreview.url}
            style={{ display: 'block', maxHeight: `${HOVER_PREVIEW_MAX_PX}px`, maxWidth: `${HOVER_PREVIEW_MAX_PX}px` }}
          />
        </Box>
      ) : null}
    </Box>
  );
};

export default ImageMapPlot;
