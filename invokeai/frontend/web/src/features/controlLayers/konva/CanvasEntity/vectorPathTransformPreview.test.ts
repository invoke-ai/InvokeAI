import Konva from 'konva/lib/index.js';
import { describe, expect, it } from 'vitest';

import { prepareVectorPathTransformPreview } from './vectorPathTransformPreview';

describe('prepareVectorPathTransformPreview', () => {
  it('keeps other paths in a static preview group', () => {
    const objectGroup = new Konva.Group({ x: 12, y: 20, listening: false });
    const activePath = new Konva.Path({ name: 'object_renderer:vector_path:active', data: 'M 0 0 L 10 0' });
    const otherPath = new Konva.Path({ name: 'object_renderer:vector_path:other', data: 'M 0 10 L 10 10' });
    objectGroup.add(activePath, otherPath);

    const result = prepareVectorPathTransformPreview(objectGroup, 'active');

    expect(result).not.toBeNull();
    expect(activePath.visible()).toBe(true);
    expect(otherPath.visible()).toBe(false);
    expect(result?.previewGroup.getChildren().map((node) => node.name())).toEqual([
      'object_renderer:vector_path:other',
    ]);
    expect(result?.previewGroup.x()).toBe(12);

    objectGroup.x(40);
    expect(result?.previewGroup.x()).toBe(12);
  });

  it('does not change visibility when the active path is missing', () => {
    const objectGroup = new Konva.Group();
    const path = new Konva.Path({ name: 'object_renderer:vector_path:other', data: 'M 0 0 L 10 0' });
    objectGroup.add(path);

    expect(prepareVectorPathTransformPreview(objectGroup, 'missing')).toBeNull();
    expect(path.visible()).toBe(true);
  });
});
