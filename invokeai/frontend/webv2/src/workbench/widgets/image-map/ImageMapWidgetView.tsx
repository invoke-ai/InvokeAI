import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Button, Center, Spinner, Stack, Text } from '@chakra-ui/react';
import { getImageMapClickSelectsCluster } from '@workbench/image-map/imageMapSettings';
import { ensureImageMapLoaded, imageMapStore, refreshImageMapPoints } from '@workbench/image-map/imageMapStore';
import { useWidgetValuesSelector } from '@workbench/WorkbenchContext';
import { lazy, Suspense, useEffect } from 'react';

// Lazy so the plotly bundle (its own vite chunk, ~1.5MB) loads only when the
// widget is actually shown.
const ImageMapPlot = lazy(() => import('./ImageMapPlot'));

const handleRefresh = () => {
  void refreshImageMapPoints();
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
  const { data, error, loadState, renderError } = imageMapStore.useSnapshot();
  const clickSelectsCluster = useWidgetValuesSelector('image-map', getImageMapClickSelectsCluster);

  useEffect(() => {
    ensureImageMapLoaded();
  }, []);

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
    // also sits above `loadWidget`, which preloads only the implementation
    // chunk and cannot reach this nested import, so a preset switch onto an
    // already-loaded map suspended anyway. Confining it here keeps the frame
    // mounted and the spinner where the plot will appear.
    return (
      <Suspense fallback={plotLoadingFallback}>
        <ImageMapPlot clickSelectsCluster={clickSelectsCluster} />
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

  if (data?.state === 'model_missing') {
    const model = data.modelName ?? 'the model named by `image_index_model`';

    return (
      <CenteredMessage
        detail={`Image indexing is enabled, but the embedding model '${model}' is not installed. Install the CLIP Vision model with this name, then restart the server.`}
        title="Embedding model not installed"
      />
    );
  }

  if (!data || data.state === 'disabled') {
    return (
      <CenteredMessage
        detail="Enable `image_index_enabled` in the server configuration and restart the server to build a semantic index of your gallery."
        title="Image indexing is off"
      />
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
  onAction,
  title,
}: {
  title: string;
  detail: string;
  actionLabel?: string;
  onAction?: () => void;
}) => (
  <Center h="full" p="6">
    <Stack align="center" gap="2" maxW="sm" textAlign="center">
      <Text fontWeight="semibold">{title}</Text>
      <Text color="fg.muted" fontSize="sm">
        {detail}
      </Text>
      {actionLabel && onAction ? (
        <Button mt="2" onClick={onAction} size="xs" variant="outline">
          {actionLabel}
        </Button>
      ) : null}
    </Stack>
  </Center>
);
