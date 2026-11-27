export interface QueueProgressImage {
  dataUrl: string;
  height: number;
  width: number;
}

export const mergeQueueProgressImage = <Image extends QueueProgressImage>(
  previous: Image | null | undefined,
  next: Image | null | undefined
): Image | null | undefined => (next === undefined ? previous : next);
