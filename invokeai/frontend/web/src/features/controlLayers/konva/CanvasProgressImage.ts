import Konva from 'konva';

export class CanvasProgressImage {
  id: string;
  progressImageId: string | null;
  konvaImageGroup: Konva.Group;
  konvaImage: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  isLoading: boolean;
  isError: boolean;

  constructor(arg: { id: string }) {
    const { id } = arg;
    this.konvaImageGroup = new Konva.Group({ id, listening: false });

    this.id = id;
    this.progressImageId = null;
    this.konvaImage = null;
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
    const imageEl = new Image();
    imageEl.onload = () => {
      if (this.konvaImage) {
        this.konvaImage.setAttrs({
          image: imageEl,
          x,
          y,
          width,
          height,
        });
      } else {
        this.konvaImage = new Konva.Image({
          id: this.id,
          listening: false,
          image: imageEl,
          x,
          y,
          width,
          height,
        });
        this.konvaImageGroup.add(this.konvaImage);
      }
    };
    imageEl.id = progressImageId;
    imageEl.src = dataURL;
  }

  destroy() {
    this.konvaImageGroup.destroy();
  }
}
