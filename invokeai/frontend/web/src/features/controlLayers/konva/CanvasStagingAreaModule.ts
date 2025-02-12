import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasStagingAreaSlice, selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import type { StagingAreaImage } from 'features/controlLayers/store/types';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import Konva from 'konva';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';

export class CanvasStagingAreaModule extends CanvasModuleBase {
  readonly type = 'staging_area';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  subscriptions: Set<() => void> = new Set();
  konva: { group: Konva.Group };
  image: CanvasObjectImage | null;
  selectedImage: StagingAreaImage | null;

  $shouldShowStagedImage = atom<boolean>(true);
  $isStaging = atom<boolean>(false);

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.konva = { group: new Konva.Group({ name: `${this.type}:group`, listening: false }) };
    this.image = null;
    this.selectedImage = null;

    /**
     * When we change this flag, we need to re-render the staging area, which hides or shows the staged image.
     */
    this.subscriptions.add(this.$shouldShowStagedImage.listen(this.render));
    /**
     * When the staging redux state changes (i.e. when the selected staged image is changed, or we add/discard a staged
     * image), we need to re-render the staging area.
     */
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasStagingAreaSlice, this.render));
    /**
     * Sync the $isStaging flag with the redux state. $isStaging is used by the manager to determine the global busy
     * state of the canvas.
     *
     * We also set the $shouldShowStagedImage flag when we enter staging mode, so that the staged images are shown,
     * even if the user disabled this in the last staging session.
     */
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectIsStaging, (isStaging, oldIsStaging) => {
        this.$isStaging.set(isStaging);
        if (isStaging && !oldIsStaging) {
          this.$shouldShowStagedImage.set(true);
        }
      })
    );
  }

  initialize = () => {
    this.log.debug('Initializing module');
    this.render();
    this.$isStaging.set(this.manager.stateApi.runSelector(selectIsStaging));
  };

  render = async () => {
    this.log.trace('Rendering staging area');
    const stagingArea = this.manager.stateApi.runSelector(selectCanvasStagingAreaSlice);

    const { x, y } = this.manager.stateApi.getBbox().rect;
    const shouldShowStagedImage = this.$shouldShowStagedImage.get();

    this.selectedImage = stagingArea.stagedImages[stagingArea.selectedStagedImageIndex] ?? null;
    this.konva.group.position({ x, y });

    if (this.selectedImage) {
      const { imageDTO } = this.selectedImage;
      const image = imageDTOToImageWithDims(imageDTO);

      /**
       * When the final output image of a generation is received, we should clear that generation's last progress image.
       *
       * It's possible that we have already rendered the progress image from the next generation before the output image
       * from the previous is fully loaded/rendered. This race condition results in a flicker:
       * - LAST GENERATION: Render the final progress image
       * - LAST GENERATION: Start loading the final output image...
       * - NEXT GENERATION: Render the first progress image
       * - LAST GENERATION: ...Finish loading the final output image & render it, clearing the progress image <-- Flicker!
       * - NEXT GENERATION: Render the next progress image
       *
       * We can detect the race condition by stashing the session ID of the last progress image when we begin loading
       * that session's output image. After we render it, if the progress image's session ID is the same as the one we
       * stashed, we know that we have not yet gotten that next generation's first progress image. We can clear the
       * progress image without causing a flicker.
       */
      const lastProgressEventSessionId = this.manager.progressImage.$lastProgressEvent.get()?.session_id;
      const hideProgressIfSameSession = () => {
        const currentProgressEventSessionId = this.manager.progressImage.$lastProgressEvent.get()?.session_id;
        if (lastProgressEventSessionId === currentProgressEventSessionId) {
          this.manager.progressImage.$lastProgressEvent.set(null);
        }
      };

      if (!this.image) {
        this.image = new CanvasObjectImage(
          {
            id: 'staging-area-image',
            type: 'image',
            image,
          },
          this
        );
        await this.image.update(this.image.state, true);
        this.konva.group.add(this.image.konva.group);
        hideProgressIfSameSession();
      } else if (this.image.isLoading) {
        // noop - just wait for the image to load
      } else if (this.image.state.image.image_name !== image.image_name) {
        await this.image.update({ ...this.image.state, image }, true);
        hideProgressIfSameSession();
      } else if (this.image.isError) {
        hideProgressIfSameSession();
      }
      this.image.konva.group.visible(shouldShowStagedImage);
    } else {
      this.image?.destroy();
      this.image = null;
    }
  };

  getNodes = () => {
    return [this.konva.group];
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    if (this.image) {
      this.image.destroy();
    }
    for (const node of this.getNodes()) {
      node.destroy();
    }
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      selectedImage: this.selectedImage,
      $shouldShowStagedImage: this.$shouldShowStagedImage.get(),
      $isStaging: this.$isStaging.get(),
      image: this.image?.repr() ?? null,
    };
  };
}
