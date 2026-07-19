export type StreamingImageKind = 'fallback' | 'final' | 'live';

export interface DenoiseProgressImage {
  dataUrl: string;
  height: number;
  width: number;
}

export interface StreamingImageSource {
  alt: string;
  height?: number;
  kind: StreamingImageKind;
  src: string;
  width?: number;
}

export const mergeDenoiseProgressImage = <Image extends DenoiseProgressImage>(
  previous: Image | null | undefined,
  next: Image | null | undefined
): Image | null | undefined => (next === undefined ? previous : next);

export const progressImageToStreamingSource = (
  image: DenoiseProgressImage | null | undefined,
  alt = 'In-progress diffusion preview'
): StreamingImageSource | null =>
  image
    ? {
        alt,
        height: image.height,
        kind: 'live',
        src: image.dataUrl,
        width: image.width,
      }
    : null;

export const imageUrlToStreamingSource = ({
  alt,
  height,
  kind = 'final',
  src,
  width,
}: {
  alt: string;
  height?: number;
  kind?: Exclude<StreamingImageKind, 'live'>;
  src: string | null | undefined;
  width?: number;
}): StreamingImageSource | null =>
  src
    ? {
        alt,
        height,
        kind,
        src,
        width,
      }
    : null;

export const resolveStreamingImageSource = ({
  fallbackImage,
  finalImage,
  heldLiveImage,
  liveImage,
}: {
  fallbackImage?: StreamingImageSource | null;
  finalImage?: StreamingImageSource | null;
  heldLiveImage?: StreamingImageSource | null;
  liveImage?: StreamingImageSource | null;
}): StreamingImageSource | null => finalImage ?? liveImage ?? heldLiveImage ?? fallbackImage ?? null;
