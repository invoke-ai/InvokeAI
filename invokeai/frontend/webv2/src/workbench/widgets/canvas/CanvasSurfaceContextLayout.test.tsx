import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { CanvasSurfaceContextLayout } from './CanvasSurfaceContextLayout';

const engineSurface = <span data-engine-surface="">Surface</span>;
const noop = () => undefined;

const renderLayout = (surface: ReactNode): string =>
  renderToStaticMarkup(
    <ChakraProvider value={system}>
      <div data-canvas-widget="">
        <CanvasSurfaceContextLayout surface={surface} onContextMenu={noop}>
          <div data-canvas-chrome="">Chrome</div>
        </CanvasSurfaceContextLayout>
      </div>
    </ChakraProvider>
  );

describe('CanvasSurfaceContextLayout', () => {
  it('keeps one empty context-menu owner mounted beside chrome without an engine surface', () => {
    const markup = renderLayout(null);

    expect(markup.match(/data-canvas-context-menu-owner/g)).toHaveLength(1);
    expect(markup).toMatch(/data-canvas-context-menu-owner=""[^>]*><\/div><div data-canvas-chrome="">Chrome<\/div>/);
  });

  it('contains the engine surface inside the owner while chrome remains its sibling', () => {
    const markup = renderLayout(engineSurface);

    expect(markup.match(/data-canvas-context-menu-owner/g)).toHaveLength(1);
    expect(markup).toMatch(
      /data-canvas-context-menu-owner=""[^>]*><span data-engine-surface="">Surface<\/span><\/div><div data-canvas-chrome="">Chrome<\/div>/
    );
  });
});
