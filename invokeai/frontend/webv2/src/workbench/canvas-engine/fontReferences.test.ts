import { describe, expect, it } from 'vitest';

import type { CanvasLayerContract, CanvasTextFontRef } from './contracts';

import { documentFrom, groupContract, layerContract } from './document-model/documentFixtures.testStub';
import { collectCanvasFontReferences, replaceCanvasFontReferences, sameCanvasTextFontRef } from './fontReferences';

const from: CanvasTextFontRef = {
  contentHash: 'a'.repeat(64),
  family: 'Catalog Sans',
  id: 'font-a',
  label: 'Catalog Sans Regular',
};
const to: CanvasTextFontRef = {
  contentHash: 'b'.repeat(64),
  family: 'Replacement Sans',
  id: 'font-b',
  label: 'Replacement Sans Variable',
};

const textLayer = (
  id: string,
  fontRef: CanvasTextFontRef,
  fontVariations?: Readonly<Record<string, number>>
): CanvasLayerContract =>
  layerContract(id, 'raster', {
    source: {
      align: 'left',
      color: '#ffffff',
      content: id,
      fontFamily: fontRef.family,
      fontRef,
      fontSize: 24,
      fontVariations,
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text',
    },
  });

describe('canvas font references', () => {
  it('groups exact custom references across stacks and nested groups', () => {
    const other: CanvasTextFontRef = { ...from, label: 'Catalog Sans Italic' };
    const document = documentFrom([
      groupContract('group', [textLayer('nested-a', from), textLayer('nested-b', from)]),
      textLayer('root-other', other),
    ]);
    document.stacks.control.push(textLayer('control-a', from));

    expect(collectCanvasFontReferences(document)).toEqual([
      { count: 3, fontRef: from, layerIds: ['nested-a', 'nested-b', 'control-a'] },
      { count: 1, fontRef: other, layerIds: ['root-other'] },
    ]);
  });

  it('replaces every exact match, clamps supported axes, and reports resets', () => {
    const document = documentFrom([
      groupContract('group', [
        textLayer('clamped', from, { opsz: 14, wght: 900 }),
        textLayer('reset', from, { ital: 1, wght: 250 }),
      ]),
      textLayer('untouched', { ...from, label: 'Other face' }, { wght: 900 }),
    ]);

    const result = replaceCanvasFontReferences(document, from, {
      fontRef: to,
      style: 'normal',
      weight: 400,
      axes: [{ default: 400, maximum: 700, minimum: 300, tag: 'wght' }],
    });

    expect(result.replacedLayerIds).toEqual(['clamped', 'reset']);
    expect(result.replacedCount).toBe(2);
    expect(result.clampedAxisTags).toEqual(['wght']);
    expect(result.resetAxisTags).toEqual(['ital', 'opsz']);
    expect(result.document.version).toBe(4);
    const group = result.document.stacks.raster[0];
    expect(group?.type).toBe('group');
    if (group?.type === 'group') {
      expect(group.children[0]).toMatchObject({
        source: {
          fontFamily: to.family,
          fontRef: to,
          fontVariations: { wght: 700 },
          fontWeight: 700,
          fontStyle: 'normal',
        },
      });
      expect(group.children[1]).toMatchObject({
        source: {
          fontFamily: to.family,
          fontRef: to,
          fontVariations: { wght: 300 },
          fontWeight: 300,
          fontStyle: 'normal',
        },
      });
    }
    expect(result.document.stacks.raster[1]).toEqual(document.stacks.raster[1]);
  });

  it('resets all coordinates for a static replacement and treats same-ref replacement as a no-op', () => {
    const document = documentFrom([textLayer('text', from, { opsz: 14, wght: 500 })]);
    const staticResult = replaceCanvasFontReferences(document, from, {
      fontRef: to,
      style: 'italic',
      weight: 700,
      axes: [],
    });
    const source = staticResult.document.stacks.raster[0];
    expect(source?.type === 'raster' ? source.source : null).toMatchObject({
      fontRef: to,
      fontFamily: to.family,
      fontWeight: 700,
      fontStyle: 'italic',
    });
    expect(
      source?.type === 'raster' && source.source.type === 'text' ? source.source.fontVariations : null
    ).toBeUndefined();
    expect(staticResult.resetAxisTags).toEqual(['opsz', 'wght']);

    const noOp = replaceCanvasFontReferences(document, from, { fontRef: from, style: 'normal', weight: 400, axes: [] });
    expect(noOp).toMatchObject({ replacedCount: 0, document });
    expect(sameCanvasTextFontRef(from, { ...from })).toBe(true);
  });
  it('uses target axis defaults when replacing a static face and synchronizes CSS typography', () => {
    const document = documentFrom([textLayer('text', from)]);
    const result = replaceCanvasFontReferences(document, from, {
      fontRef: to,
      style: 'normal',
      weight: 400,
      axes: [
        { tag: 'wght', minimum: 100, maximum: 900, default: 650 },
        { tag: 'ital', minimum: 0, maximum: 1, default: 1 },
      ],
    });
    expect(result.document.stacks.raster[0]).toMatchObject({
      source: { fontWeight: 650, fontStyle: 'italic', fontVariations: { wght: 650, ital: 1 }, fontRef: to },
    });
  });
});
