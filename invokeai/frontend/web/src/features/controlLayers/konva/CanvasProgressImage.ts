import { loadImage } from 'features/controlLayers/konva/util';
import Konva from 'konva';

export class CanvasProgressImage {
  static NAME_PREFIX = 'progress-image';
  static GROUP_NAME = `${CanvasProgressImage.NAME_PREFIX}_group`;
  static IMAGE_NAME = `${CanvasProgressImage.NAME_PREFIX}_image`;

  id: string;
  progressImageId: string | null;
  konva: {
    group: Konva.Group;
    image: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  };
  isLoading: boolean;
  isError: boolean;

  constructor(arg: { id: string }) {
    const { id } = arg;
    this.konva = {
      group: new Konva.Group({ name: CanvasProgressImage.GROUP_NAME, listening: false }),
      image: null,
    };
    this.id = id;
    this.progressImageId = null;
    this.isLoading = false;
    this.isError = false;
  }

  async updateImageSource(
    progressImageId: string,
    dataURL: string,
    x: number,
    y: number,
    width: number,
    height: number
  ) {
    if (this.isLoading) {
      return;
    }
    this.isLoading = true;
    try {
      const imageEl = await loadImage(dataURL);
      if (this.konva.image) {
        this.konva.image.setAttrs({
          image: imageEl,
          x,
          y,
          width,
          height,
        });
      } else {
        this.konva.image = new Konva.Image({
          name: CanvasProgressImage.IMAGE_NAME,
          listening: false,
          image: imageEl,
          x,
          y,
          width,
          height,
        });
        this.konva.group.add(this.konva.image);
      }
      this.isLoading = false;
      this.id = progressImageId;
    } catch {
      this.isError = true;
    }
  }

  destroy() {
    this.konva.group.destroy();
  }
}
