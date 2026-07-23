import type { Rect } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import type { Ideogram4RegionInput } from './buildIdeogram4Prompt';
import { buildIdeogram4Caption, rectToIdeogram4Bbox } from './buildIdeogram4Prompt';

describe('rectToIdeogram4Bbox', () => {
  const genBbox: Rect = { x: 0, y: 0, width: 1024, height: 1024 };

  it('normalizes a region rect to [y_min, x_min, y_max, x_max] in 0-1000', () => {
    // Region occupying the right half, vertically centered band.
    const regionRect: Rect = { x: 512, y: 256, width: 512, height: 512 };
    expect(rectToIdeogram4Bbox(regionRect, genBbox)).toEqual([250, 500, 750, 1000]);
  });

  it('accounts for a generation bbox not anchored at the origin', () => {
    const offsetBbox: Rect = { x: 100, y: 200, width: 1000, height: 500 };
    const regionRect: Rect = { x: 600, y: 200, width: 500, height: 250 };
    // x: (600-100)/1000=0.5 -> 500, (1100-100)/1000=1.0 -> 1000
    // y: (200-200)/500=0 -> 0, (450-200)/500=0.5 -> 500
    expect(rectToIdeogram4Bbox(regionRect, offsetBbox)).toEqual([0, 500, 500, 1000]);
  });

  it('clamps out-of-bounds regions to the 0-1000 range', () => {
    const regionRect: Rect = { x: -200, y: -200, width: 2048, height: 2048 };
    expect(rectToIdeogram4Bbox(regionRect, genBbox)).toEqual([0, 0, 1000, 1000]);
  });

  it('returns zeros for a degenerate (zero-size) generation bbox', () => {
    const regionRect: Rect = { x: 10, y: 10, width: 10, height: 10 };
    expect(rectToIdeogram4Bbox(regionRect, { x: 0, y: 0, width: 0, height: 0 })).toEqual([0, 0, 0, 0]);
  });
});

describe('buildIdeogram4Caption', () => {
  const region = (prompt: string, bbox: Ideogram4RegionInput['bbox']): Ideogram4RegionInput => ({ prompt, bbox });

  it('passes through a raw JSON object verbatim and marks it structured', () => {
    const raw = '{"high_level_description":"a cat","compositional_deconstruction":{"background":"","elements":[]}}';
    expect(buildIdeogram4Caption(raw, [])).toEqual({ prompt: raw, isStructured: true });
  });

  it('passes through raw JSON even when leading/trailing whitespace is present', () => {
    const raw = '  {"a":1}  ';
    expect(buildIdeogram4Caption(raw, [])).toEqual({ prompt: raw, isStructured: true });
  });

  it('returns plain text (not structured) when there are no regions', () => {
    expect(buildIdeogram4Caption('a golden retriever on a skateboard', [])).toEqual({
      prompt: 'a golden retriever on a skateboard',
      isStructured: false,
    });
  });

  it('trims the plain prompt', () => {
    expect(buildIdeogram4Caption('  hello world  ', [])).toEqual({ prompt: 'hello world', isStructured: false });
  });

  it('assembles a structured caption from regions with correct key order', () => {
    const result = buildIdeogram4Caption('A dog and a ball.', [
      region('a fluffy dog', [200, 300, 800, 900]),
      region('a red ball', [250, 750, 750, 950]),
    ]);
    expect(result.isStructured).toBe(true);
    // Key order must be exactly: high_level_description, compositional_deconstruction{background, elements};
    // each obj element: type, bbox, desc. JSON.stringify preserves insertion order, so assert on the raw string.
    expect(result.prompt).toBe(
      '{"high_level_description":"A dog and a ball.",' +
        '"compositional_deconstruction":{"background":"",' +
        '"elements":[' +
        '{"type":"obj","bbox":[200,300,800,900],"desc":"a fluffy dog"},' +
        '{"type":"obj","bbox":[250,750,750,950],"desc":"a red ball"}' +
        ']}}'
    );
  });

  it('omits bbox for regions with no drawn content', () => {
    const result = buildIdeogram4Caption('scene', [region('floating element', null)]);
    expect(result.prompt).toContain('{"type":"obj","desc":"floating element"}');
  });

  it('skips regions with empty prompts', () => {
    const result = buildIdeogram4Caption('scene', [region('   ', [0, 0, 100, 100]), region('kept', [1, 2, 3, 4])]);
    const parsed = JSON.parse(result.prompt);
    expect(parsed.compositional_deconstruction.elements).toHaveLength(1);
    expect(parsed.compositional_deconstruction.elements[0].desc).toBe('kept');
  });

  it('preserves non-ASCII characters without escaping', () => {
    const result = buildIdeogram4Caption('café', [region('a café ☕', [0, 0, 100, 100])]);
    expect(result.prompt).toContain('café ☕');
    expect(result.prompt).not.toContain('\\u');
  });

  it('injects a color palette as style_description.color_palette with correct key order', () => {
    const result = buildIdeogram4Caption('a sunset', [region('a boat', [0, 0, 500, 500])], ['#FF6B35', '#004E89']);
    expect(result.isStructured).toBe(true);
    expect(result.prompt).toBe(
      '{"high_level_description":"a sunset",' +
        '"style_description":{"color_palette":["#FF6B35","#004E89"]},' +
        '"compositional_deconstruction":{"background":"","elements":[{"type":"obj","bbox":[0,0,500,500],"desc":"a boat"}]}}'
    );
  });

  it('builds a structured caption from a palette alone (no regions)', () => {
    const result = buildIdeogram4Caption('a sunset', [], ['#FF6B35']);
    expect(result.isStructured).toBe(true);
    const parsed = JSON.parse(result.prompt);
    expect(parsed.style_description.color_palette).toEqual(['#FF6B35']);
    expect(parsed.compositional_deconstruction.elements).toEqual([]);
  });

  it('uppercases palette colors and drops invalid hex (shorthand, names)', () => {
    const result = buildIdeogram4Caption('x', [], ['#ff6b35', 'red', '#FFF', '#00cc88']);
    const parsed = JSON.parse(result.prompt);
    expect(parsed.style_description.color_palette).toEqual(['#FF6B35', '#00CC88']);
  });

  it('ignores the palette for raw-JSON passthrough', () => {
    const raw = '{"a":1}';
    expect(buildIdeogram4Caption(raw, [], ['#FF6B35'])).toEqual({ prompt: raw, isStructured: true });
  });

  it('stays plain text when there are no regions and no palette', () => {
    expect(buildIdeogram4Caption('just text', [])).toEqual({ prompt: 'just text', isStructured: false });
  });
});
