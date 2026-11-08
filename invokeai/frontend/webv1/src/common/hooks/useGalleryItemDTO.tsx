import { isVideoName } from 'features/gallery/store/types';
import { useImageDTO } from 'services/api/endpoints/images';
import { useVideoDTO } from 'services/api/endpoints/videos';
import type { ImageDTO, VideoDTO } from 'services/api/types';

/**
 * Resolves either an ImageDTO or a VideoDTO based on a polymorphic name. The kind is derived
 * from the filename extension — the backend names images `<uuid>.png` and videos `<uuid>.mp4`,
 * so we can dispatch without an extra fetch.
 *
 * Both underlying RTK Query hooks are always called (React rule-of-hooks), but only the relevant
 * one is given a real name; the other receives `null` and short-circuits via `skipToken`.
 */
/** @knipignore Re-exported for callers that destructure the hook return into named locals. */
export type GalleryItemDTO = { kind: 'image'; dto: ImageDTO } | { kind: 'video'; dto: VideoDTO };

export const useGalleryItemDTO = (name: string | null | undefined): GalleryItemDTO | null => {
  const isVideo = name ? isVideoName(name) : false;
  const imageName = name && !isVideo ? name : null;
  const videoName = name && isVideo ? name : null;

  const imageDTO = useImageDTO(imageName);
  const videoDTO = useVideoDTO(videoName);

  if (!name) {
    return null;
  }
  if (isVideo) {
    return videoDTO ? { kind: 'video', dto: videoDTO } : null;
  }
  return imageDTO ? { kind: 'image', dto: imageDTO } : null;
};
