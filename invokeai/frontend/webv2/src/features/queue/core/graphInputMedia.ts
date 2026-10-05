/**
 * Exclude compiled input media from results: primitive nodes echo inputs, while generated outputs receive fresh
 * server names.
 */

export interface GraphInputMediaNames {
  imageNames: ReadonlySet<string>;
  videoNames: ReadonlySet<string>;
}

/** Collect image_name/video_name strings from compiled nodes; malformed persisted graphs yield empty sets. */
export const collectGraphInputMediaNames = (graph: unknown): GraphInputMediaNames => {
  const imageNames = new Set<string>();
  const videoNames = new Set<string>();

  const visit = (value: unknown): void => {
    if (Array.isArray(value)) {
      for (const entry of value) {
        visit(entry);
      }
      return;
    }
    if (!value || typeof value !== 'object') {
      return;
    }

    const imageName = (value as { image_name?: unknown }).image_name;
    if (typeof imageName === 'string') {
      imageNames.add(imageName);
    }
    const videoName = (value as { video_name?: unknown }).video_name;
    if (typeof videoName === 'string') {
      videoNames.add(videoName);
    }

    for (const entry of Object.values(value)) {
      visit(entry);
    }
  };

  const nodes = graph && typeof graph === 'object' ? (graph as { nodes?: unknown }).nodes : undefined;
  if (nodes && typeof nodes === 'object') {
    for (const node of Object.values(nodes)) {
      visit(node);
    }
  }

  return { imageNames, videoNames };
};
