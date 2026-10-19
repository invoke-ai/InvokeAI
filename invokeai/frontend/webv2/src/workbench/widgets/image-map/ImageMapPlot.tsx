import type { GalleryItemKey } from '@features/gallery/contracts';
import type { ImageMapPoint } from '@workbench/image-map/api';
import type { ClusterAnnotation, PointAppearance } from '@workbench/image-map/imageMapTraces';
import type { AxisRanges } from '@workbench/image-map/imageMapViewport';
import type { HoverCluster, HoverPreview } from '@workbench/widgets/image-map/MapHoverCard';
import type { PlotlyHTMLElement } from 'plotly.js';

import { Box } from '@chakra-ui/react';
import {
  getImageCluster,
  getPersistedSelectedGalleryItemKeys,
  getSelectedGalleryItemFromValues,
  parseGallerySemanticReference,
  toGalleryItemKey,
} from '@features/gallery/contracts';
import { attachWheelZoom } from '@workbench/image-map/attachWheelZoom';
import { collectClusterSelection, formatClusterSize } from '@workbench/image-map/clusterSelection';
import { imageMapStore } from '@workbench/image-map/imageMapStore';
import {
  ALL_POINTS_TRACE,
  buildAllPointsTraces,
  buildClusterAnnotations,
  buildClusterSelectionTraces,
  buildCurrentImageTrace,
  buildHighlightedPointsTrace,
  buildMapLayout,
  CLUSTER_SELECTION_IMAGES_TRACE,
  CLUSTER_SELECTION_VIDEOS_TRACE,
  CURRENT_IMAGE_TRACE,
  declutterAnnotations,
  getTraceAppearance,
  HIGHLIGHTED_POINTS_TRACE,
  toBaseAppearanceRestyle,
  toClusterSelectionRestyle,
  toHighlightRestyle,
} from '@workbench/image-map/imageMapTraces';
import {
  computePercentileRanges,
  expandRangesToInclude,
  fitRangesToAspect,
  rangesToKeepMarkerInView,
} from '@workbench/image-map/imageMapViewport';
import { getThumbnailUrl } from '@workbench/image-map/thumbnailCache';
import { MapHoverCard } from '@workbench/widgets/image-map/MapHoverCard';
import { shallowEqual, useWidgetValuesSelector } from '@workbench/WorkbenchContext';
import Plotly from 'plotly.js-gl2d-dist-min';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

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
 * Fit the percentile box plus current marker once the container is measured, preserving equal axis scale so a
 * zero-size initial fit cannot become the permanent view.
 */
const computeInitialFit = (
  points: ImageMapPoint[],
  selectedKey: GalleryItemKey | null,
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

  const selected = selectedKey ? points.find((candidate) => candidate.key === selectedKey) : undefined;

  if (selected) {
    box = expandRangesToInclude(box, selected);
  }

  return fitRangesToAspect(box, width / height);
};

const findTraceIndex = (plot: PlotElement, name: string): number =>
  (plot.data ?? []).findIndex((trace) => (trace as { name?: string }).name === name);

/** Lazy-load the imperative Plotly host; Plotly owns its DOM outside JSX. */
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
  // Require matching clustering for hover tags: refresh may renumber ids even while stale annotations remain
  // briefly visible.
  const clusterLabelsMatchPoints = imageMapStore.useSelector(
    (snapshot) =>
      snapshot.clusterLabelsHash !== null &&
      snapshot.clusterLabelsHash === snapshot.data?.visibleHash &&
      snapshot.clusterLabelsEps === snapshot.data?.clusterEps
  );
  // Use item keys for selection/recentering equality, including videos.
  const selectedKey = useWidgetValuesSelector('gallery', (values) => {
    const item = getSelectedGalleryItemFromValues(values);

    return item ? toGalleryItemKey({ kind: item.kind, name: item.name }) : null;
  });
  const selectedItemKeys = useWidgetValuesSelector(
    'gallery',
    (values) => getPersistedSelectedGalleryItemKeys(values),
    shallowEqual
  );
  // With a cluster filter active in the gallery, the whole cluster stays lit
  // on a dimmed map — the gallery is showing exactly these images.
  const clusterQueryItemKeys = useWidgetValuesSelector(
    'gallery',
    (values) => {
      const reference = parseGallerySemanticReference(values.semanticImageQuery);

      return reference?.kind === 'cluster' ? (getImageCluster(reference.clusterId)?.itemKeys ?? null) : null;
    },
    shallowEqual
  );
  const selectedKeys = useMemo(() => new Set(selectedItemKeys), [selectedItemKeys]);
  const clusterKeys = useMemo(
    () => (clusterQueryItemKeys ? new Set(clusterQueryItemKeys) : null),
    [clusterQueryItemKeys]
  );
  const { selectCluster, selectItem: selectMapItem } = useMapSelection();
  // Bumped after every scene rebuild so the overlay effects (marker,
  // highlight) re-apply onto the fresh, empty overlay traces.
  const [plotRevision, setPlotRevision] = useState(0);
  const lastPinchAtRef = useRef(0);
  const lastMapSelectionRef = useRef<{ name: string; at: number } | null>(null);
  // The selection the marker effect last decluttered for. Seeded with the
  // mounting selection so the initial scene, whose labels the effect below
  // applies from scratch, does not also get a redundant pass here.
  const lastDeclutteredSelectionRef = useRef(selectedKey);
  // The points array `fullAnnotationsRef` was built from, so the marker effect
  // can tell whether those annotations describe the embedding now on screen.
  const annotationsPointsRef = useRef<ImageMapPoint[] | null>(null);
  // Coordinate scene and resize paths to perform whole-map fitting once, after real measurement.
  const initialFitDoneRef = useRef(false);
  const pointsRef = useRef(points);
  const selectedKeyRef = useRef(selectedKey);
  const clusterModeRef = useRef(clickSelectsCluster);
  const clusterLabelsRef = useRef(clusterLabels);

  // Declared before the effects below so the refs are fresh when they run.
  useEffect(() => {
    pointsRef.current = points;
    selectedKeyRef.current = selectedKey;
    clusterModeRef.current = clickSelectsCluster;
    clusterLabelsRef.current = clusterLabels;
  }, [clickSelectsCluster, clusterLabels, points, selectedKey]);
  // Keep full annotations in a ref so relayout can refilter visibility without React renders.
  const fullAnnotationsRef = useRef<ClusterAnnotation[]>([]);
  // The base traces' own looks for the scene on screen, so dimming can be
  // undone exactly; replaced with every rebuild, which also resets the dim.
  const baseAppearancesRef = useRef<PointAppearance[]>([]);
  // Whether the base traces on screen are currently dimmed. A rebuild draws
  // them undimmed, so it resets this too.
  const baseDimmedRef = useRef(false);
  const appliedAnnotationsKeyRef = useRef<string | null>(null);

  // Declutter against the current view and skip unchanged visibility to avoid relayout feedback. Reserve space for
  // the gold target beneath Plotly's annotation layer.
  const applyDeclutteredAnnotations = useCallback((container: PlotElement) => {
    const ranges = readRanges(container);
    const selectedItemKey = selectedKeyRef.current;
    const markerPoint =
      (selectedItemKey ? pointsRef.current?.find((candidate) => candidate.key === selectedItemKey) : null) ?? null;
    const annotations =
      ranges && container.offsetWidth > 0 && container.offsetHeight > 0
        ? declutterAnnotations(
            fullAnnotationsRef.current,
            ranges,
            container.offsetWidth,
            container.offsetHeight,
            markerPoint
          )
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

    // The base points come back as several traces, one per appearance; the
    // overlays are found by name, so their position among them does not
    // matter. That lookup is load-bearing now in a way it was not when the
    // base was a single trace at a fixed index: it is safe because
    // `Plotly.react` swaps `gd.data` synchronously, so the overlay effects
    // that run straight after this one already see the new indices. Overlay traces (highlight, marker) start empty; the overlay
    // effects below restyle them, so a selection change never rebuilds the
    // scene.
    const baseTraces = buildAllPointsTraces(points);
    const traces = [
      ...baseTraces,
      // Above the base points, below the hand-selection outline and the gold
      // target, which both mark individual items within a lit cluster.
      ...buildClusterSelectionTraces(points, new Set()),
      buildHighlightedPointsTrace(points, new Set()),
      buildCurrentImageTrace(),
    ];
    baseAppearancesRef.current = baseTraces.map(getTraceAppearance);
    baseDimmedRef.current = false;
    let disposed = false;

    // Preserve user ranges across refreshes, but use an aspect-corrected fit for the first measured render rather
    // than retaining zero-size artifacts.
    let initialRanges = readRanges(container as unknown as PlotElement) ?? computePercentileRanges(points);

    if (!initialFitDoneRef.current) {
      const fitted = computeInitialFit(points, selectedKeyRef.current, container.offsetWidth, container.offsetHeight);

      if (fitted) {
        initialRanges = fitted;
        initialFitDoneRef.current = true;
      }
    }

    // Apply annotations via relayout, not scene reconstruction. Clear their applied key when rebuilding layout so
    // identical labels are restored.
    appliedAnnotationsKeyRef.current = null;
    const layout = buildMapLayout(initialRanges);

    void Plotly.react(container, traces as Plotly.Data[], layout, {
      displayModeBar: false,
      // Use custom wheel/pinch zoom to avoid Plotly scrollZoom issues on Safari.
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

          const customdata = event.points?.[0]?.customdata;
          // Resolved against the points this scene was built from rather than
          // parsed: customdata is this component's own key, so one that names
          // no point is a stale event, not an item to go and fetch.
          const clicked = typeof customdata === 'string' ? points.find((point) => point.key === customdata) : undefined;

          if (!clicked) {
            return;
          }

          // A selection made by clicking the map must not recenter the map
          // under the user's cursor; the marker effect checks this. The
          // stamp expires so a stale entry (failed hydrate, re-click of the
          // current point) cannot suppress a legitimate future recenter.
          lastMapSelectionRef.current = { at: Date.now(), name: clicked.key };

          const clusterKeys = clusterModeRef.current ? collectClusterSelection(points, clicked.key) : null;

          if (clusterKeys) {
            // Use the primary cluster phrase for filter chips, falling back to member count; alternates belong in
            // hover cards.
            const label =
              clusterLabelsRef.current?.[String(clicked.cluster)]?.label ?? formatClusterSize(clusterKeys.length);

            selectCluster(clicked.item, clusterKeys, label);
          } else {
            // Also the cluster-mode fallback for noise points (cluster -1).
            selectMapItem(clicked.item);
          }
        });
        plot.removeAllListeners?.('plotly_hover');
        plot.on('plotly_hover', (event) => {
          const customdata = event.points?.[0]?.customdata;
          const mouse = (event as { event?: MouseEvent }).event;
          const hovered = typeof customdata === 'string' ? points.find((point) => point.key === customdata) : undefined;

          if (!hovered || !mouse) {
            return;
          }

          clearHover();
          const session = hoverSessionRef.current;
          const { clientX, clientY } = mouse;
          hoverTimerRef.current = setTimeout(() => {
            void getThumbnailUrl(hovered.item).then((thumbnail) => {
              // Deliberately not gated on `disposed`: that belongs to this
              // effect, which re-runs on every socket-driven refresh, so a
              // refresh landing mid-dwell would drop the thumbnail — and no
              // new `plotly_hover` fires while the pointer sits still.
              // Unmount is covered by the session bump in the cleanup below.
              if (thumbnail && hoverSessionRef.current === session) {
                setHoverPreview({ clientX, clientY, key: hovered.key, thumbnail });
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

        // Report WebGL initialization as renderError so valid points cannot repeatedly remount the failing plot.
        imageMapStore.patchSnapshot({ renderError: 'The map failed to render (WebGL unavailable).' });
      });

    return () => {
      disposed = true;
    };
  }, [applyDeclutteredAnnotations, points, selectCluster, selectMapItem]);

  // Cluster selection: the members redrawn at full colour over a dimmed map.
  // Restyled in place like the other overlays; the base traces take one
  // scalar per trace, so neither the click nor later zooms pay per point.
  useEffect(() => {
    const container = containerRef.current as PlotElement | null;

    if (!container || points === null) {
      return;
    }

    const imagesIndex = findTraceIndex(container, CLUSTER_SELECTION_IMAGES_TRACE);
    const videosIndex = findTraceIndex(container, CLUSTER_SELECTION_VIDEOS_TRACE);

    if (imagesIndex < 0 || videosIndex < 0) {
      return;
    }

    const traces = buildClusterSelectionTraces(points, clusterKeys ?? new Set());
    // A cluster none of whose members are on this map (all since deleted)
    // lights nothing, so it must not grey everything out either.
    const dimmed = traces.some((trace) => trace.x.length > 0);
    const baseIndices = (container.data ?? []).flatMap((trace, index) =>
      (trace as { name?: string }).name === ALL_POINTS_TRACE ? [index] : []
    );

    swallow(Plotly.restyle(container, toClusterSelectionRestyle(traces), [imagesIndex, videosIndex]));

    // Restyling the base redraws every point, so only a change of state pays for it.
    if (dimmed !== baseDimmedRef.current && baseIndices.length === baseAppearancesRef.current.length) {
      baseDimmedRef.current = dimmed;
      swallow(Plotly.restyle(container, toBaseAppearanceRestyle(baseAppearancesRef.current, dimmed), baseIndices));
    }
  }, [clusterKeys, plotRevision, points]);

  // Highlight overlay: the gallery's hand-made multi-selection, restyled in place.
  useEffect(() => {
    const container = containerRef.current as PlotElement | null;

    if (!container || points === null) {
      return;
    }

    const highlightIndex = findTraceIndex(container, HIGHLIGHTED_POINTS_TRACE);

    if (highlightIndex < 0) {
      return;
    }

    const trace = buildHighlightedPointsTrace(points, selectedKeys);
    swallow(Plotly.restyle(container, toHighlightRestyle(trace), [highlightIndex]));
  }, [plotRevision, points, selectedKeys]);

  // Dispose the timer at component lifetime, not on every points update.
  useEffect(() => clearHover, []);

  // Retire hover sessions when items leave; hiding alone could resurrect stale pointer coordinates. Recenter the
  // live selected marker near view edges while preserving zoom.
  useEffect(() => {
    const container = containerRef.current as PlotElement | null;

    if (!container || points === null) {
      return;
    }

    const markerIndex = findTraceIndex(container, CURRENT_IMAGE_TRACE);

    if (markerIndex < 0) {
      return;
    }

    // Redeclutter selection moves only with annotations matching current points; refresh effects rebuild
    // annotations separately, including batched selection/refresh commits.
    const selectionChanged =
      lastDeclutteredSelectionRef.current !== selectedKey && annotationsPointsRef.current === points;
    lastDeclutteredSelectionRef.current = selectedKey;

    const point = selectedKey ? points.find((candidate) => candidate.key === selectedKey) : undefined;
    const suppression = lastMapSelectionRef.current;
    const isSuppressionFresh = suppression !== null && Date.now() - suppression.at < MAP_CLICK_SUPPRESS_MS;
    const cameFromMapClick = isSuppressionFresh && suppression.name === selectedKey;

    // Consume matching suppression tickets and expire abandoned ones before missing-point returns; unrelated
    // selections must not consume a pending click's ticket.
    if (suppression !== null && (cameFromMapClick || !isSuppressionFresh)) {
      lastMapSelectionRef.current = null;
    }

    if (!point) {
      swallow(Plotly.restyle(container, { x: [[]], y: [[]] }, [markerIndex]));

      if (selectionChanged) {
        // Labels dropped to clear the old marker position get their spot back.
        applyDeclutteredAnnotations(container);
      }

      return;
    }

    swallow(Plotly.restyle(container, { x: [[point.x]], y: [[point.y]] }, [markerIndex]));

    if (selectionChanged) {
      // Redeclutter after marker moves to hide newly covered labels and restore uncovered ones, even without
      // recentering.
      applyDeclutteredAnnotations(container);
    }

    if (cameFromMapClick) {
      return;
    }

    const ranges = readRanges(container);
    const recentered = ranges ? rangesToKeepMarkerInView(ranges, point) : null;

    if (recentered) {
      swallow(Plotly.relayout(container, { 'xaxis.range': recentered.x, 'yaxis.range': recentered.y }));
    }
  }, [applyDeclutteredAnnotations, plotRevision, points, selectedKey]);

  // Apply delayed labels via relayout so coordinate arrays and selection traces remain intact; hover cards receive
  // full alternate-tag data separately.
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
    annotationsPointsRef.current = points;
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
              selectedKeyRef.current,
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
      // Reset the fit flag on purge because its layout no longer exists, including StrictMode remounts.
      initialFitDoneRef.current = false;
    };
  }, []);

  // Keep hover across refreshes only while its item remains in live points. Null points on account invalidation
  // must also remove the old thumbnail.
  const hoverPreview =
    pendingHoverPreview && points?.some((point) => point.key === pendingHoverPreview.key) ? pendingHoverPreview : null;

  // Resolve cluster identity, size, and color from live points so stationary hover follows renumbered clusters.
  const hoverCluster = useMemo((): HoverCluster => {
    if (!hoverPreview || !points) {
      return { cluster: -1, clusterSize: 0 };
    }

    const cluster = points.find((point) => point.key === hoverPreview.key)?.cluster ?? -1;

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
          key={hoverPreview.key}
          preview={hoverPreview}
        />
      ) : null}
    </Box>
  );
};

export default ImageMapPlot;
