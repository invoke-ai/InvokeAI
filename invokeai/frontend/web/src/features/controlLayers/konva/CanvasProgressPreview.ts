import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasProgressImage } from 'features/controlLayers/konva/CanvasProgressImage';
import Konva from 'konva';
import type { InvocationDenoiseProgressEvent } from 'services/events/types';

export class CanvasProgressPreview {
  group: Konva.Group;
  progressImage: CanvasProgressImage;
  manager: CanvasManager;

  constructor(manager: CanvasManager) {
    this.manager = manager;
    this.group = new Konva.Group({ listening: false });
    this.progressImage = new CanvasProgressImage({ id: 'progress-image' });
    this.group.add(this.progressImage.konvaImageGroup);
  }

  async render(lastProgressEvent: InvocationDenoiseProgressEvent | null) {
    const bboxRect = this.manager.stateApi.getBbox().rect;

    if (lastProgressEvent) {
      const { invocation, step, progress_image } = lastProgressEvent;
      const { dataURL } = progress_image;
      const { x, y, width, height } = bboxRect;
      const progressImageId = `${invocation.id}_${step}`;
      if (
        !this.progressImage.isLoading &&
        !this.progressImage.isError &&
        this.progressImage.progressImageId !== progressImageId
      ) {
        await this.progressImage.updateImageSource(progressImageId, dataURL, x, y, width, height);
        this.progressImage.konvaImageGroup.visible(true);
      }
    } else {
      this.progressImage.konvaImageGroup.visible(false);
    }
  }
}
