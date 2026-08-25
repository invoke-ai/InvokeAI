import type { ImageMapClusterLabelInfo, ImageMapImageLabels, ImageMapPoint } from '@workbench/image-map/api';
import type { ClusterAnnotation } from '@workbench/image-map/imageMapTraces';
import type { AxisRanges } from '@workbench/image-map/imageMapViewport';
import type { PlotlyHTMLElement } from 'plotly.js';
import type { CSSProperties } from 'react';

import { Box } from '@chakra-ui/react';
import {
  getImageCluster,
  getPersistedSelectedGalleryItemKeys,
  getSelectedGalleryImageFromValues,
  parseGalleryItemKey,
  parseGallerySemanticReference,
} from '@features/gallery/contracts';
import { attachWheelZoom } from '@workbench/image-map/attachWheelZoom';
import { getClusterColor, isClusterColorLight } from '@workbench/image-map/clusterPalette';
import { collectClusterSelection } from '@workbench/image-map/clusterSelection';
import { getImageLabels } from '@workbench/image-map/imageLabelCache';
import { imageMapStore } from '@workbench/image-map/imageMapStore';
import {
  buildAllPointsTrace,
  buildClusterAnnotations,
  buildCurrentImageTrace,
  buildHighlightedPointsTrace,
  buildMapLayout,
  CURRENT_IMAGE_TRACE,
  declutterAnnotations,
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
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

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

/** Bounds the hover card's thumbnail. */
const HOVER_PREVIEW_MAX_PX = 160;
const HOVER_PREVIEW_OFFSET_PX = 14;
/** Keep the hover card at least this clear of the viewport edges. */
const HOVER_PREVIEW_EDGE_PAD_PX = 10;

interface HoverPreview {
  imageName: string;
  url: string;
  clientX: number;
  clientY: number;
}

/**
 * Cluster identity for the hovered image, resolved from the CURRENT points.
 * Deliberately not captured at hover time: a hover survives a live refresh
 * (see `hoverPreview` below), and a refresh re-runs DBSCAN, which can renumber
 * every cluster. A frozen id would then be paired with the new clustering's
 * labels and color — a card describing a cluster the image is not in.
 */
interface HoverCluster {
  /** DBSCAN cluster of the hovered point; -1 means unclustered noise. */
  cluster: number;
  /** Points currently on the map in that cluster. */
  clusterSize: number;
}

const FIRST_TAG_STYLE: CSSProperties = { fontStyle: 'italic', fontWeight: 'bold' };
const REST_TAG_STYLE: CSSProperties = { fontStyle: 'italic' };
const HOVER_IMG_STYLE: CSSProperties = {
  borderRadius: '6px',
  display: 'block',
  margin: '0 auto',
  maxHeight: `${HOVER_PREVIEW_MAX_PX}px`,
  maxWidth: `${HOVER_PREVIEW_MAX_PX}px`,
};

/** "a, b, c" with the first tag emphasized, all on the cluster color. */
const HoverTagsRow = ({ prefix, tags, style }: { prefix: string; tags: string[]; style: CSSProperties }) => (
  <Box fontSize="xs" px="2" py="0.5" style={style} textAlign="center">
    {prefix}
    {tags.map((tag, index) => (
      <span key={tag} style={index === 0 ? FIRST_TAG_STYLE : REST_TAG_STYLE}>
        {index > 0 ? ', ' : ''}
        {tag}
      </span>
    ))}
  </Box>
);

/**
 * The hover card: thumbnail, filename, cluster identity/size, and the top
 * cluster and image tags — PhotoMapAI's popup. The card is tinted with the
 * hovered cluster's color, and the text flips dark/light to stay readable on
 * it. Its size depends on async content (the thumbnail and the lazily
 * fetched image tags), so it renders invisibly, is measured, and is then
 * placed beside the cursor — flipped to the other side when it would leave
 * the viewport. Parents key this by image name so a new hover starts clean.
 */
const MapHoverCard = ({
  preview,
  hoverCluster,
  clusterLabel,
}: {
  preview: HoverPreview;
  hoverCluster: HoverCluster;
  clusterLabel: ImageMapClusterLabelInfo | null;
}) => {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [imageLabels, setImageLabels] = useState<ImageMapImageLabels | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);

  // Image tags are computed on demand (network round-trip on first hover of
  // an image; session-cached after). The card renders without the row until
  // they arrive.
  useEffect(() => {
    let cancelled = false;

    void getImageLabels(preview.imageName).then((labels) => {
      if (!cancelled) {
        setImageLabels(labels);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [preview.imageName]);

  // Re-measure whenever content that changes the card's size lands.
  useLayoutEffect(() => {
    const card = cardRef.current;

    if (!card) {
      return;
    }

    const rect = card.getBoundingClientRect();
    let left = preview.clientX + HOVER_PREVIEW_OFFSET_PX;
    let top = preview.clientY + HOVER_PREVIEW_OFFSET_PX;

    if (left + rect.width > window.innerWidth - HOVER_PREVIEW_EDGE_PAD_PX) {
      left = Math.max(0, preview.clientX - rect.width - HOVER_PREVIEW_OFFSET_PX);
    }

    if (top + rect.height > window.innerHeight - HOVER_PREVIEW_EDGE_PAD_PX) {
      top = Math.max(0, preview.clientY - rect.height - HOVER_PREVIEW_OFFSET_PX);
    }

    setPosition({ left, top });
  }, [preview.clientX, preview.clientY, imageLabels, imageLoaded, clusterLabel, hoverCluster]);

  // The palette color drives every style on the card; memoized so JSX gets
  // stable objects (and text stays readable via the dark/light flip).
  const styles = useMemo(() => {
    const clusterColor = getClusterColor(hoverCluster.cluster);
    const lightBackground = isClusterColorLight(clusterColor);
    const color = lightBackground ? '#222222' : '#FFFFFF';
    const textShadow = lightBackground ? '0 1px 2px #FFFFFF' : '0 1px 2px #000000';

    return {
      band: { background: 'rgba(0, 0, 0, 0.25)', color, textShadow } satisfies CSSProperties,
      card: { background: clusterColor, border: `2px solid ${clusterColor}` } satisfies CSSProperties,
      filename: { color, textShadow, wordBreak: 'break-all' } satisfies CSSProperties,
      tags: { color, textShadow } satisfies CSSProperties,
    };
  }, [hoverCluster.cluster]);

  const clusterTags = clusterLabel ? [clusterLabel.label, ...clusterLabel.alternates].slice(0, 3) : null;
  const imageTags = imageLabels ? [imageLabels.label, ...imageLabels.alternates].slice(0, 3) : null;

  return (
    <Box
      left={`${position?.left ?? 0}px`}
      maxW="60"
      p="2"
      pb="1"
      pointerEvents="none"
      position="fixed"
      ref={cardRef}
      rounded="lg"
      shadow="lg"
      style={styles.card}
      top={`${position?.top ?? 0}px`}
      visibility={position ? 'visible' : 'hidden'}
      zIndex="tooltip"
    >
      <img
        alt={preview.imageName}
        onError={() => setImageLoaded(true)}
        onLoad={() => setImageLoaded(true)}
        src={preview.url}
        style={HOVER_IMG_STYLE}
      />
      <Box fontSize="xs" mt="1" style={styles.filename} textAlign="center">
        {preview.imageName}
      </Box>
      <Box fontSize="xs" fontWeight="bold" mt="1" py="0.5" rounded="sm" style={styles.band} textAlign="center">
        {hoverCluster.cluster < 0
          ? 'Unclustered'
          : `Cluster ${hoverCluster.cluster} (size=${hoverCluster.clusterSize})`}
      </Box>
      {clusterTags ? <HoverTagsRow prefix="Cluster tags: " style={styles.tags} tags={clusterTags} /> : null}
      {imageTags ? <HoverTagsRow prefix="Image tags: " style={styles.tags} tags={imageTags} /> : null}
    </Box>
  );
};

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
  // Whether those labels were computed over the clustering now drawn. The
  // annotations accept a stale set for the ~1s until fresh ones land (see the
  // relayout effect below), but the hover card names one specific cluster by
  // id, and a refresh can renumber every id — so it shows no tags rather than
  // another cluster's.
  const clusterLabelsMatchPoints = imageMapStore.useSelector(
    (snapshot) => snapshot.clusterLabelsHash !== null && snapshot.clusterLabelsHash === snapshot.data?.visibleHash
  );
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
  // With a cluster filter active in the gallery, the whole cluster stays
  // highlighted on the map — the gallery is showing exactly these images.
  const clusterQueryImageNames = useWidgetValuesSelector(
    'gallery',
    (values) => {
      const reference = parseGallerySemanticReference(values.semanticImageQuery);

      return reference?.kind === 'cluster' ? (getImageCluster(reference.clusterId)?.imageNames ?? null) : null;
    },
    shallowEqual
  );
  const selectedNames = useMemo(
    () => new Set([...selectedImageNames, ...(clusterQueryImageNames ?? [])]),
    [clusterQueryImageNames, selectedImageNames]
  );
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
  const clusterLabelsRef = useRef(clusterLabels);

  // Declared before the effects below so the refs are fresh when they run.
  useEffect(() => {
    pointsRef.current = points;
    selectedImageNameRef.current = selectedImageName;
    clusterModeRef.current = clickSelectsCluster;
    clusterLabelsRef.current = clusterLabels;
  }, [clickSelectsCluster, clusterLabels, points, selectedImageName]);
  // The full annotation set for the current data; which of them actually show
  // is view-dependent (see applyDeclutteredAnnotations), so the source list
  // lives in a ref the relayout listener can re-filter without re-rendering.
  const fullAnnotationsRef = useRef<ClusterAnnotation[]>([]);
  const appliedAnnotationsKeyRef = useRef<string | null>(null);

  // Zoomed far out, cluster labels pile onto the same few pixels; declutter
  // against the CURRENT view so only labels with room to breathe render, and
  // zooming back in restores the rest. The applied-key check makes the common
  // case (a pan/zoom that changes no label's visibility) a no-op — it also
  // keeps this from feeding back into itself through the plotly_relayout
  // event its own relayout fires.
  const applyDeclutteredAnnotations = useCallback((container: PlotElement) => {
    const ranges = readRanges(container);
    const annotations =
      ranges && container.offsetWidth > 0 && container.offsetHeight > 0
        ? declutterAnnotations(fullAnnotationsRef.current, ranges, container.offsetWidth, container.offsetHeight)
        : fullAnnotationsRef.current;
    const key = annotations.map((annotation) => `${annotation.text}@${annotation.x},${annotation.y}`).join('\n');

    if (key === appliedAnnotationsKeyRef.current) {
      return;
    }

    appliedAnnotationsKeyRef.current = key;
    swallow(Plotly.relayout(container, { annotations }));
  }, []);
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

    // Annotations are deliberately absent here: they are a layout concern, and
    // rebuilding the scene for them is what made labels arriving a second after
    // the points re-materialize every trace. See the relayout effect below.
    // Since the rebuilt layout carries no annotations, the applied-key must be
    // forgotten or the label effect would skip re-adding an identical set.
    appliedAnnotationsKeyRef.current = null;
    const layout = buildMapLayout(initialRanges);

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
            const clicked = points.find((point) => point.imageName === imageName);
            // The backend's cluster label names the filter chip when one has
            // arrived; the member count is the fallback (labels can be off,
            // still loading, or missing for this cluster). Only the primary
            // phrase is used — the alternates belong to the hover card, which
            // has room to show them.
            const label =
              (clicked ? clusterLabelsRef.current?.[String(clicked.cluster)]?.label : undefined) ??
              `${clusterNames.length} images`;

            selectCluster(imageName, clusterNames, label);
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
        plot.removeAllListeners?.('plotly_relayout');
        plot.on('plotly_relayout', (event) => {
          // Every zoom, pan, and resize changes which labels have room; the
          // annotation-only relayouts this triggers carry no axis keys, so
          // they fall through without recursing.
          const viewChanged = Object.keys(event ?? {}).some(
            (key) => key.startsWith('xaxis.') || key.startsWith('yaxis.') || key === 'autosize'
          );

          if (viewChanged) {
            applyDeclutteredAnnotations(container as unknown as PlotElement);
          }
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
  }, [applyDeclutteredAnnotations, points, selectCluster, selectImage]);

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

  // Labels arrive about a second after the points they annotate. Applying them
  // with `relayout` rather than through the scene effect keeps that from
  // re-materializing every coordinate array — and, more visibly, from resetting
  // the highlight and current-image traces to empty, which made the gold marker
  // and the multi-select highlight blink off and back on with every refresh.
  // The annotation builder wants the display strings only; the full label
  // info (alternates included) feeds the hover card below.
  const annotationLabels = useMemo(
    () =>
      clusterLabels === null
        ? null
        : Object.fromEntries(Object.entries(clusterLabels).map(([clusterId, info]) => [clusterId, info.label])),
    [clusterLabels]
  );

  useEffect(() => {
    const container = containerRef.current as PlotElement | null;

    if (!container || points === null || !container.data) {
      return;
    }

    fullAnnotationsRef.current = buildClusterAnnotations(points, showClusterLabels ? annotationLabels : null);
    applyDeclutteredAnnotations(container);
  }, [annotationLabels, applyDeclutteredAnnotations, plotRevision, points, showClusterLabels]);

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
  // Requires a live point set, not just one that does not contradict the
  // hover: `points` goes null when the account is invalidated, and the card
  // must go with it rather than keep the previous account's thumbnail on
  // screen — with a cluster identity that resolves to the noise sentinel.
  const hoverPreview =
    pendingHoverPreview && points?.some((point) => point.imageName === pendingHoverPreview.imageName)
      ? pendingHoverPreview
      : null;

  // Resolved from the live points, so the id, the size and the tint describe
  // the clustering currently drawn even after a refresh has renumbered it
  // under a stationary pointer. `hoverPreview` non-null already implies the
  // image is in `points`, so there is no missing-point fallback to take.
  const hoverCluster = useMemo((): HoverCluster => {
    if (!hoverPreview || !points) {
      return { cluster: -1, clusterSize: 0 };
    }

    const cluster = points.find((point) => point.imageName === hoverPreview.imageName)?.cluster ?? -1;

    return {
      cluster,
      clusterSize: points.reduce((count, point) => (point.cluster === cluster ? count + 1 : count), 0),
    };
  }, [hoverPreview, points]);

  return (
    <Box h="full" minH="0" position="relative" w="full">
      <Box ref={containerRef} h="full" w="full" />
      {hoverPreview ? (
        <MapHoverCard
          clusterLabel={(clusterLabelsMatchPoints ? clusterLabels?.[String(hoverCluster.cluster)] : null) ?? null}
          hoverCluster={hoverCluster}
          key={hoverPreview.imageName}
          preview={hoverPreview}
        />
      ) : null}
    </Box>
  );
};

export default ImageMapPlot;
