import type Konva from 'konva';

export const prepareVectorPathTransformPreview = (objectGroup: Konva.Group, pathId: string) => {
  const activePathNode = objectGroup.getChildren().find((node) => node.name().endsWith(`:${pathId}`));
  if (!activePathNode) {
    return null;
  }

  objectGroup.clearCache();
  const previewGroup = objectGroup.clone({ name: 'vector_path_transform_preview', listening: false });
  for (const node of previewGroup.getChildren()) {
    if (node.name().endsWith(`:${pathId}`)) {
      node.destroy();
    }
  }

  for (const node of objectGroup.getChildren()) {
    node.visible(node === activePathNode);
  }

  return { activePathNode, previewGroup };
};
