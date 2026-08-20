import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Button, Center, Spinner, Stack, Text } from '@chakra-ui/react';
import { getImageMapClickSelectsCluster, getImageMapShowClusterLabels } from '@workbench/image-map/imageMapSettings';
import {
  ensureImageMapLoaded,
  imageMapStore,
  refreshImageIndexStatus,
  refreshImageMapPoints,
  setClusterLabelsEnabled,
} from '@workbench/image-map/imageMapStore';
import { isIndexing } from '@workbench/image-map/indexProgress';
import { useWidgetValuesSelector } from '@workbench/WorkbenchContext';
import { lazy, Suspense, useEffect } from 'react';

import { ImageIndexProgressPanel } from './ImageIndexProgress';

// Lazy so the plotly bundle (its own vite chunk, ~1.5MB) loads only when the
// widget is actually shown.
const ImageMapPlot = lazy(() => import('./ImageMapPlot'));

const handleRefresh = () => {
  void refreshImageMapPoints();
  // The counts too: when the retry is the progress panel's, stale counts are
  // the likeliest reason the user pressed it.
  refreshImageIndexStatus();
};

// Module scope: an inline element would be a new value on every render, which
// is both what react-perf/jsx-no-jsx-as-prop forbids and pointless here — the
// fallback never varies.
const plotLoadingFallback = (
  <Center h="full">
    <Spinner size="lg" />
  </Center>
);

/**
 * Semantic map of the gallery: every image embedded by the backend's image
 * index, projected to 2D with UMAP and colored by cluster. Clicking a point
 * selects that image in the gallery (and so in Preview).
 */
export const ImageMapWidgetView = (_props: WidgetViewProps) => {
  const { data, error, indexCounts, indexUpdatedAt, loadState, renderError } = imageMapStore.useSnapshot();
  const clickSelectsCluster = useWidgetValuesSelector('image-map', getImageMapClickSelectsCluster);
  const showClusterLabels = useWidgetValuesSelector('image-map', getImageMapShowClusterLabels);

  useEffect(() => {
    ensureImageMapLoaded();
  }, []);

  // Pushed into the store so turning labels off stops the request, not just the
  // drawing of what it returns.
  useEffect(() => {
    setClusterLabelsEnabled(showClusterLabels);
  }, [showClusterLabels]);

  // Checked before the plot: this is the canvas failing, not a fetch, so
  // re-mounting the plot would just fail again and render an empty box with no
  // way out. A successful refresh clears it and lets the plot retry.
  if (renderError) {
    return (
      <CenteredMessage
        actionLabel="Retry"
        detail={renderError}
        onAction={handleRefresh}
        title="Image map unavailable"
      />
    );
  }

  // A working map beats a full-screen error: when a refresh fails but prior
  // points exist, keep showing them (the next successful refresh recovers).
  if (data && data.points.length > 0) {
    // Its own boundary, rather than leaning on WidgetRenderer's. That one wraps
    // the whole widget, so suspending on the plotly chunk replaced the entire
    // panel — header and actions menu included — with a skeleton frame, and
    // then held the resolved content for React's fallback throttle on top. It
    // also sits above `loadWidgets`, which preloads only the implementation
    // chunk and cannot reach this nested import, so a preset switch onto an
    // already-loaded map suspended anyway. Confining it here keeps the frame
    // mounted and the spinner where the plot will appear.
    return (
      <Suspense fallback={plotLoadingFallback}>
        <ImageMapPlot clickSelectsCluster={clickSelectsCluster} showClusterLabels={showClusterLabels} />
      </Suspense>
    );
  }

  if (loadState === 'idle' || loadState === 'loading') {
    return (
      <Center h="full">
        <Spinner size="lg" />
      </Center>
    );
  }

  if (loadState === 'error' && !data) {
    return (
      <CenteredMessage
        actionLabel="Retry"
        detail={error ?? 'Failed to load the image map.'}
        onAction={handleRefresh}
        title="Image map unavailable"
      />
    );
  }

  const retainedDataError = loadState === 'error' ? (error ?? 'Failed to load the image map.') : null;

  if (data?.state === 'model_missing') {
    const model = data.modelName ?? 'the model named by `image_index_model`';

    return (
      <CenteredMessage
        actionLabel={retainedDataError ? 'Retry' : undefined}
        detail={`Image indexing is enabled, but the embedding model '${model}' is not installed. Install the CLIP Vision model with this name, then restart the server.`}
        errorDetail={retainedDataError}
        onAction={retainedDataError ? handleRefresh : undefined}
        title="Embedding model not installed"
      />
    );
  }

  if (!data || data.state === 'disabled') {
    return (
      <CenteredMessage
        actionLabel={retainedDataError ? 'Retry' : undefined}
        detail="Enable `image_index_enabled` in the server configuration and restart the server to build a semantic index of your gallery."
        errorDetail={retainedDataError}
        onAction={retainedDataError ? handleRefresh : undefined}
        title="Image indexing is off"
      />
    );
  }

  // Ahead of both `computing` and the empty state: with nothing to draw yet,
  // how far the backfill has got is the one thing that answers "when will
  // there be a map?" — a spinner or "nothing to map yet" leaves a user with a
  // large gallery unable to tell progress from a stall. `computing` is routine
  // here (the backend asks for a projection as soon as anything is embedded),
  // so deferring to it would hide the progress for most of a backfill.
  //
  // A failed refresh is carried into the panel rather than shadowed by it:
  // preempting the `error` branch below would otherwise drop both the message
  // and the only retry the widget has before the map is ready.
  if (isIndexing(indexCounts)) {
    return (
      <Center h="full" p="6">
        <ImageIndexProgressPanel
          counts={indexCounts}
          error={loadState === 'error' ? (error ?? 'Failed to load the image map.') : null}
          updatedAt={indexUpdatedAt}
          onRetry={handleRefresh}
        />
      </Center>
    );
  }

  if (data.state === 'computing') {
    return (
      <Center h="full">
        <Stack align="center" gap="3">
          <Spinner size="lg" />
          <Text color="fg.muted" fontSize="sm">
            Computing your image map…
          </Text>
          <Button onClick={handleRefresh} size="xs" variant="outline">
            Check again
          </Button>
        </Stack>
      </Center>
    );
  }

  if (loadState === 'error') {
    return (
      <CenteredMessage
        actionLabel="Retry"
        detail={error ?? 'Failed to load the image map.'}
        onAction={handleRefresh}
        title="Image map unavailable"
      />
    );
  }

  return (
    <CenteredMessage
      detail="Generate or import images and they will appear here, clustered by visual similarity."
      title="Nothing to map yet"
    />
  );
};

const CenteredMessage = ({
  actionLabel,
  detail,
  errorDetail,
  onAction,
  title,
}: {
  title: string;
  detail: string;
  actionLabel?: string;
  errorDetail?: string | null;
  onAction?: () => void;
}) => (
  <Center h="full" p="6">
    <Stack align="center" gap="2" maxW="sm" textAlign="center">
      <Text fontWeight="semibold">{title}</Text>
      <Text color="fg.muted" fontSize="sm">
        {detail}
      </Text>
      {errorDetail ? (
        <Text color="fg.error" fontSize="sm" maxW="full" minW="0" overflowWrap="anywhere" role="alert">
          {errorDetail}
        </Text>
      ) : null}
      {actionLabel && onAction ? (
        <Button mt="2" onClick={onAction} size="xs" variant="outline">
          {actionLabel}
        </Button>
      ) : null}
    </Stack>
  </Center>
);
