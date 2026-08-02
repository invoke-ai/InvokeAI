import type { PlotlyHTMLElement } from 'plotly.js';

import { Box } from '@chakra-ui/react';
import { imageMapStore } from '@workbench/image-map/imageMapStore';
import { buildAllPointsTrace, buildCurrentImageTrace, buildMapLayout } from '@workbench/image-map/imageMapTraces';
import Plotly from 'plotly.js-gl2d-dist-min';
import { useEffect, useRef } from 'react';

import { useSelectMapImage } from './useSelectMapImage';

/**
 * Imperative plotly host. All plotly calls happen in effects against a ref
 * div — plotly manages its own DOM and must never render through JSX. This
 * module is lazy-loaded so the plotly bundle stays out of the app's critical
 * path.
 */
const ImageMapPlot = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const points = imageMapStore.useSelector((snapshot) => snapshot.data?.points ?? null);
  const selectImage = useSelectMapImage();

  useEffect(() => {
    const container = containerRef.current;

    if (!container || points === null) {
      return;
    }

    const traces = [buildAllPointsTrace(points), buildCurrentImageTrace()];
    let disposed = false;

    void Plotly.react(container, traces as Plotly.Data[], buildMapLayout(), {
      displayModeBar: false,
      // Wheel/pinch zoom is replaced with custom handlers in a later PR;
      // plotly's own scrollZoom has long-standing Safari issues.
      scrollZoom: false,
    })
      .then((plot: PlotlyHTMLElement) => {
        if (disposed) {
          return;
        }

        plot.removeAllListeners?.('plotly_click');
        plot.on('plotly_click', (event) => {
          const imageName = event.points?.[0]?.customdata;

          if (typeof imageName === 'string') {
            selectImage(imageName);
          }
        });
      })
      .catch(() => {
        // WebGL context creation can fail (blocked GPU, context exhaustion);
        // an unhandled rejection would bypass the widget failure boundary, so
        // surface it through the store's error state instead.
        imageMapStore.patchSnapshot({ error: 'The map failed to render (WebGL unavailable).', loadState: 'error' });
      });

    return () => {
      disposed = true;
    };
  }, [points, selectImage]);

  // Track the widget frame's size; plotly does not observe its container.
  useEffect(() => {
    const container = containerRef.current;

    if (!container) {
      return;
    }

    const observer = new ResizeObserver(() => {
      if (container.offsetWidth > 0 && container.offsetHeight > 0) {
        void Plotly.Plots.resize(container);
      }
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      Plotly.purge(container);
    };
  }, []);

  return <Box ref={containerRef} h="full" minH="0" w="full" />;
};

export default ImageMapPlot;
