/** Dependency-neutral result returned after persisting or staging canvas pixels. */
export interface CanvasImageUploadResult {
  imageName: string;
  width: number;
  height: number;
}
