/**
 * Media names referenced by a compiled graph's node field values — the run's INPUTS
 * (keyframes, source videos, reference images).
 *
 * Settlement collects result media by walking every node output in the backend
 * session. A media primitive node echoes its input into `session.results` under the
 * same name (e.g. the first-frame keyframe of an image-to-video workflow), so without
 * this set the runtime would treat the user's source image as a generated result:
 * board-attach it and record it on every run. Generated outputs are always saved
 * server-side under fresh names, so a result name that appears in the compiled graph
 * can only be an input passthrough.
 */

export interface GraphInputMediaNames {
  imageNames: ReadonlySet<string>;
  videoNames: ReadonlySet<string>;
}

/**
 * Collect every `image_name` / `video_name` string reachable from the compiled
 * graph's nodes. `graph` is `unknown` because it comes from persisted snapshots;
 * anything malformed yields empty sets (previous routing behavior).
 */
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
