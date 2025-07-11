import { Mutex } from 'async-mutex';
import type { ProgressData, ProgressDataMap } from 'features/controlLayers/components/SimpleSession/context';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasImageState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Atom } from 'nanostores';
import { atom, effect } from 'nanostores';
import type { Logger } from 'roarr';
import type { S } from 'services/api/types';

// To get pixel sizes corresponding to our theme tokens, first find the theme token CSS var in browser dev tools.
// For example `var(--invoke-space-8)` is equivalent to using `8` as a space prop in a component.
//
// If it is already in pixels, you can use it directly. If it is in rems, you need to convert it to pixels.
//
// For example:
// const style = window.getComputedStyle(document.documentElement)
// parseFloat(style.fontSize) * parseFloat(style.getPropertyValue("--invoke-space-8"))
//
// This will give you the pixel value for the theme token in pixels.
//
// You cannot do this dynamically in this file, because it depends on the styles being applied to the document, which
// will not have happened yet when this module is loaded.

const SPACING_4 = 12; // --invoke-space-4 in pixels
const BORDER_RADIUS_BASE = 4; // --invoke-radii-base in pixels
const BORDER_WIDTH = 1;
const FONT_SIZE_MD = 14.4; // --invoke-fontSizes-md
const BADGE_WIDTH = 192;
const BADGE_HEIGHT = 36;

type ImageNameSrc = { type: 'imageName'; data: string };
type DataURLSrc = { type: 'dataURL'; data: string };

export class CanvasStagingAreaModule extends CanvasModuleBase {
  readonly type = 'staging_area';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  subscriptions: Set<() => void> = new Set();
  konva: {
    group: Konva.Group;
    placeholder: {
      group: Konva.Group;
      badgeBg: Konva.Rect;
      text: Konva.Text;
    };
  };
  image: CanvasObjectImage | null;
  mutex = new Mutex();

  $imageSrc = atom<ImageNameSrc | DataURLSrc | null>(null);

  $shouldShowStagedImage = atom<boolean>(true);
  $isStaging = atom<boolean>(false);
  $isPending = atom<boolean>(false);

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.konva = {
      group: new Konva.Group({
        name: `${this.type}:group`,
        listening: false,
      }),
      placeholder: {
        group: new Konva.Group({
          name: `${this.type}:placeholder_group`,
          listening: false,
          visible: false,
        }),
        badgeBg: new Konva.Rect({
          name: `${this.type}:placeholder_badge_bg`,
          fill: 'hsl(220 12% 10% / 0.8)', // 'base.900' with opacity
          x: SPACING_4,
          y: SPACING_4,
          width: BADGE_WIDTH,
          height: BADGE_HEIGHT,
          cornerRadius: BORDER_RADIUS_BASE,
          stroke: 'hsl(220 12% 50% / 1)', // 'base.700'
          strokeWidth: BORDER_WIDTH,
          listening: false,
          perfectDrawEnabled: false,
        }),
        text: new Konva.Text({
          name: `${this.type}:placeholder_text`,
          fill: 'hsl(220 12% 80% / 1)', // 'base.300'
          x: SPACING_4,
          y: SPACING_4,
          width: BADGE_WIDTH,
          height: BADGE_HEIGHT,
          align: 'center',
          verticalAlign: 'middle',
          fontFamily: '"Inter Variable", sans-serif',
          fontSize: FONT_SIZE_MD,
          fontStyle: '600', // Equivalent to theme fontWeight "semibold"
          text: 'Waiting for Image',
          listening: false,
          perfectDrawEnabled: false,
        }),
      },
    };

    this.konva.placeholder.group.add(this.konva.placeholder.badgeBg);
    this.konva.placeholder.group.add(this.konva.placeholder.text);
    this.konva.group.add(this.konva.placeholder.group);

    this.image = null;

    /**
     * Rerender when the anything important changes.
     */
    this.subscriptions.add(this.$imageSrc.listen(this.render));
    this.subscriptions.add(this.$shouldShowStagedImage.listen(this.render));
    this.subscriptions.add(this.$isPending.listen(this.render));
    this.subscriptions.add(this.$isStaging.listen(this.render));

    /**
     * Sync the $isStaging flag with the redux state. $isStaging is used by the manager to determine the global busy
     * state of the canvas.
     *
     * We also set the $shouldShowStagedImage flag when we enter staging mode, so that the staged images are shown,
     * even if the user disabled this in the last staging session.
     */
    this.subscriptions.add(
      this.$isStaging.listen((isStaging, oldIsStaging) => {
        if (isStaging && !oldIsStaging) {
          this.$shouldShowStagedImage.set(true);
        }
      })
    );
  }

  initialize = () => {
    this.log.debug('Initializing module');
    this.render();
  };

  connectToSession = (
    $items: Atom<S['SessionQueueItem'][]>,
    $selectedItemId: Atom<number | null>,
    $progressData: ProgressDataMap
  ) => {
    const imageSrcListener = (
      selectedItemId: number | null,
      progressData: Record<number, ProgressData | undefined>
    ) => {
      if (!selectedItemId) {
        this.$imageSrc.set(null);
        return;
      }

      const datum = progressData[selectedItemId];

      if (datum?.imageDTO) {
        this.$imageSrc.set({ type: 'imageName', data: datum.imageDTO.image_name });
        return;
      } else if (datum?.progressImage) {
        this.$imageSrc.set({ type: 'dataURL', data: datum.progressImage.dataURL });
        return;
      } else {
        this.$imageSrc.set(null);
      }
    };
    const unsubImageSrc = effect([$selectedItemId, $progressData], imageSrcListener);

    const isPendingListener = (items: S['SessionQueueItem'][]) => {
      this.$isPending.set(items.some((item) => item.status === 'pending' || item.status === 'in_progress'));
    };
    const unsubIsPending = effect([$items], isPendingListener);

    const isStagingListener = (items: S['SessionQueueItem'][]) => {
      this.$isStaging.set(items.length > 0);
    };
    const unsubIsStaging = effect([$items], isStagingListener);

    // Run the effects & forcibly render once to initialize
    isStagingListener($items.get());
    isPendingListener($items.get());
    imageSrcListener($selectedItemId.get(), $progressData.get());
    this.render();

    return () => {
      this.$isStaging.set(false);
      unsubIsStaging();
      this.$isPending.set(false);
      unsubIsPending();
      this.$imageSrc.set(null);
      unsubImageSrc();
    };
  };

  private _getImageFromSrc = (
    { type, data }: ImageNameSrc | DataURLSrc,
    width: number,
    height: number
  ): CanvasImageState['image'] => {
    if (type === 'imageName') {
      return {
        image_name: data,
        width,
        height,
      };
    } else {
      return {
        dataURL: data,
        width,
        height,
      };
    }
  };

  render = async () => {
    const release = await this.mutex.acquire();
    try {
      this.log.trace('Rendering staging area');

      const { x, y, width, height } = this.manager.stateApi.getBbox().rect;
      const shouldShowStagedImage = this.$shouldShowStagedImage.get();
      const isPending = this.$isPending.get();

      this.konva.group.position({ x, y });

      const imageSrc = this.$imageSrc.get();

      if (imageSrc) {
        const image = this._getImageFromSrc(imageSrc, width, height);
        if (!this.image) {
          this.image = new CanvasObjectImage({ id: 'staging-area-image', type: 'image', image }, this);
          await this.image.update(this.image.state, true);
          this.konva.group.add(this.image.konva.group);
        } else if (this.image.isLoading || this.image.isError) {
          // noop
        } else {
          await this.image.update({ ...this.image.state, image });
        }
        this.konva.placeholder.group.visible(false);
      } else {
        this.image?.destroy();
        this.image = null;
        // Only show placeholder if there are pending items, otherwise show nothing
        this.konva.placeholder.group.visible(isPending);
      }

      this.konva.group.visible(shouldShowStagedImage && this.$isStaging.get());
    } finally {
      release();
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
      $shouldShowStagedImage: this.$shouldShowStagedImage.get(),
      $isStaging: this.$isStaging.get(),
      image: this.image?.repr() ?? null,
    };
  };
}
