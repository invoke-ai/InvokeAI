import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasProgressImage } from 'features/controlLayers/konva/CanvasProgressImage';
import Konva from 'konva';
import type { InvocationDenoiseProgressEvent } from 'services/events/types';

export class CanvasProgressPreview {
  static NAME_PREFIX = 'progress-preview';
  static GROUP_NAME = `${CanvasProgressPreview.NAME_PREFIX}_group`;

  konva: {
    group: Konva.Group;
    progressImage: CanvasProgressImage;
  };
  manager: CanvasManager;

  constructor(manager: CanvasManager) {
    this.manager = manager;
    this.konva = {
      group: new Konva.Group({ name: CanvasProgressPreview.GROUP_NAME, listening: false }),
      progressImage: new CanvasProgressImage({ id: 'progress-image' }),
    };
    this.konva.group.add(this.konva.progressImage.konva.group);
  }

  async render(lastProgressEvent: InvocationDenoiseProgressEvent | null) {
    const bboxRect = this.manager.stateApi.getBbox().rect;
    const session = this.manager.stateApi.getSession();

    if (lastProgressEvent && session.isStaging) {
      const { invocation, step, progress_image } = lastProgressEvent;
      const { dataURL } = progress_image;
      const { x, y, width, height } = bboxRect;
      const progressImageId = `${invocation.id}_${step}`;
      if (
        !this.konva.progressImage.isLoading &&
        !this.konva.progressImage.isError &&
        this.konva.progressImage.progressImageId !== progressImageId
      ) {
        await this.konva.progressImage.updateImageSource(progressImageId, dataURL, x, y, width, height);
        this.konva.progressImage.konva.group.visible(true);
      }
    } else {
      this.konva.progressImage.konva.group.visible(false);
    }
  }
}
