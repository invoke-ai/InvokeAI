import { Mutex } from 'async-mutex';
import type { ProgressData, ProgressDataMap } from 'features/controlLayers/components/SimpleSession/context';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import type { CanvasImageState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Atom } from 'nanostores';
import { atom, effect } from 'nanostores';
import type { Logger } from 'roarr';

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
const SPACING_2 = 6; // Equivalent to theme token 2
const SPACING_4 = 12; // Equivalent to theme token 4
const SPACING_8 = 24; // Equivalent to theme token 8
const BORDER_RADIUS_BASE = 4; // Equivalent to theme borderRadius "base"
const BORDER_WIDTH = 1; // Standard border width
const FONT_SIZE_SM = 12; // Equivalent to theme fontSize "sm"
const BADGE_MIN_WIDTH = 200;
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
      rect: Konva.Rect;
      badgeBg: Konva.Rect;
      text: Konva.Text;
    };
  };
  image: CanvasObjectImage | null;
  mutex = new Mutex();

  $imageSrc = atom<ImageNameSrc | DataURLSrc | null>(null);

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

    const { width, height } = this.manager.stateApi.getBbox().rect;

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
        rect: new Konva.Rect({
          name: `${this.type}:placeholder_rect`,
          fill: 'transparent',
          width,
          height,
          listening: false,
          perfectDrawEnabled: false,
        }),
        badgeBg: new Konva.Rect({
          name: `${this.type}:placeholder_badge_bg`,
          fill: 'hsl(220 12% 10% / 0.8)', // 'base.900' with opacity
          x: SPACING_2 - 2, // Slight offset for visual balance
          y: SPACING_2 - 2,
          width: Math.min(BADGE_MIN_WIDTH + 4, width - SPACING_2 * 2 + 4),
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
          x: SPACING_2,
          y: SPACING_2,
          width: Math.min(BADGE_MIN_WIDTH, width - SPACING_4),
          height: SPACING_8,
          align: 'center',
          verticalAlign: 'middle',
          fontFamily: '"Inter Variable", sans-serif',
          fontSize: FONT_SIZE_SM,
          fontStyle: '600', // Equivalent to theme fontWeight "semibold"
          text: 'Waiting for Image',
          listening: false,
          perfectDrawEnabled: false,
        }),
      },
    };

    this.konva.placeholder.group.add(this.konva.placeholder.rect);
    this.konva.placeholder.group.add(this.konva.placeholder.badgeBg);
    this.konva.placeholder.group.add(this.konva.placeholder.text);
    this.konva.group.add(this.konva.placeholder.group);

    this.image = null;

    /**
     * When we change this flag, we need to re-render the staging area, which hides or shows the staged image.
     */
    this.subscriptions.add(this.$shouldShowStagedImage.listen(this.render));

    /**
     * Rerender when the image source changes.
     */
    this.subscriptions.add(this.$imageSrc.listen(this.render));

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

  syncPlaceholderSize = () => {
    const { width, height } = this.manager.stateApi.getBbox().rect;
    this.konva.placeholder.rect.width(width);
    this.konva.placeholder.rect.height(height);
    this.konva.placeholder.badgeBg.width(Math.min(BADGE_MIN_WIDTH + 4, width - SPACING_2 * 2 + 4));
    this.konva.placeholder.text.width(Math.min(BADGE_MIN_WIDTH, width - SPACING_4));
  };

  initialize = () => {
    this.log.debug('Initializing module');
    this.render();
    this.$isStaging.set(this.manager.stateApi.runSelector(selectIsStaging));
  };

  connectToSession = ($selectedItemId: Atom<number | null>, $progressData: ProgressDataMap) => {
    const cb = (selectedItemId: number | null, progressData: Record<number, ProgressData | undefined>) => {
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

    // Run the effect & forcibly render once to initialize
    cb($selectedItemId.get(), $progressData.get());
    this.render();

    return effect([$selectedItemId, $progressData], cb);
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
        this.syncPlaceholderSize();
        this.konva.placeholder.group.visible(true);
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
