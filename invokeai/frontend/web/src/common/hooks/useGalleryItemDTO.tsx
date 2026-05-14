import { isCanvasProjectName, isVideoName } from 'features/gallery/store/types';
import { useGetCanvasProjectDTOQuery } from 'services/api/endpoints/canvasProjects';
import { useImageDTO } from 'services/api/endpoints/images';
import { useVideoDTO } from 'services/api/endpoints/videos';
import type { CanvasProjectDTO, ImageDTO, VideoDTO } from 'services/api/types';

/**
 * Resolves either an ImageDTO, VideoDTO or CanvasProjectDTO based on a polymorphic name. The kind
 * is derived from the filename pattern — images are `<uuid>.png`, videos `<uuid>.mp4`, canvas
 * projects bare UUIDs (no extension) — so we can dispatch without an extra fetch.
 *
 * All underlying RTK Query hooks are always called (React rule-of-hooks); only the relevant one
 * gets a real name, the others receive `null` / `skipToken` and short-circuit.
 */
/** @knipignore Re-exported for callers that destructure the hook return into named locals. */
export type GalleryItemDTO =
  | { kind: 'image'; dto: ImageDTO }
  | { kind: 'video'; dto: VideoDTO }
  | { kind: 'canvas_project'; dto: CanvasProjectDTO };

export const useGalleryItemDTO = (name: string | null | undefined): GalleryItemDTO | null => {
  const isVideo = name ? isVideoName(name) : false;
  const isCanvasProject = name ? isCanvasProjectName(name) : false;
  const imageName = name && !isVideo && !isCanvasProject ? name : null;
  const videoName = name && isVideo ? name : null;
  const projectName = name && isCanvasProject ? name : null;

  const imageDTO = useImageDTO(imageName);
  const videoDTO = useVideoDTO(videoName);
  const { data: projectDTO } = useGetCanvasProjectDTOQuery(projectName ?? '', { skip: !projectName });

  if (!name) {
    return null;
  }
  if (isCanvasProject) {
    return projectDTO ? { kind: 'canvas_project', dto: projectDTO } : null;
  }
  if (isVideo) {
    return videoDTO ? { kind: 'video', dto: videoDTO } : null;
  }
  return imageDTO ? { kind: 'image', dto: imageDTO } : null;
};
