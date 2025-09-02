import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectRuleOfThirds } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectBbox } from 'features/controlLayers/store/selectors';
import Konva from 'konva';
import type { Logger } from 'roarr';

/**
 * Renders the rule of thirds composition guide overlay on the canvas.
 * The guide shows a 3x3 grid within the bounding box to help with composition.
 */
export class CanvasCompositionGuideModule extends CanvasModuleBase {
  readonly type = 'composition_guide';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  subscriptions: Set<() => void> = new Set();

  /**
   * The Konva objects that make up the composition guide:
   * - A group to hold all the guide lines
   * - Individual line objects for the rule of thirds grid
   */
  konva: {
    group: Konva.Group;
    verticalLine1: Konva.Line;
    verticalLine2: Konva.Line;
    horizontalLine1: Konva.Line;
    horizontalLine2: Konva.Line;
  };

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating composition guide module');

    this.konva = {
      group: new Konva.Group({
        name: `${this.type}:group`,
        listening: false,
        perfectDrawEnabled: false,
      }),
      verticalLine1: new Konva.Line({
        name: `${this.type}:vertical_line_1`,
        listening: false,
        stroke: 'hsl(220 12% 90% / 0.9)',
        strokeWidth: 1,
        strokeScaleEnabled: false,
        perfectDrawEnabled: false,
        dash: [5, 5],
      }),
      verticalLine2: new Konva.Line({
        name: `${this.type}:vertical_line_2`,
        listening: false,
        stroke: 'hsl(220 12% 90% / 0.9)',
        strokeWidth: 1,
        strokeScaleEnabled: false,
        perfectDrawEnabled: false,
        dash: [5, 5],
      }),
      horizontalLine1: new Konva.Line({
        name: `${this.type}:horizontal_line_1`,
        listening: false,
        stroke: 'hsl(220 12% 90% / 0.9)',
        strokeWidth: 1,
        strokeScaleEnabled: false,
        perfectDrawEnabled: false,
        dash: [5, 5],
      }),
      horizontalLine2: new Konva.Line({
        name: `${this.type}:horizontal_line_2`,
        listening: false,
        stroke: 'hsl(220 12% 90% / 0.9)',
        strokeWidth: 1,
        strokeScaleEnabled: false,
        perfectDrawEnabled: false,
        dash: [5, 5],
      }),
    };

    this.konva.group.add(this.konva.verticalLine1);
    this.konva.group.add(this.konva.verticalLine2);
    this.konva.group.add(this.konva.horizontalLine1);
    this.konva.group.add(this.konva.horizontalLine2);

    // Listen for changes to the rule of thirds guide setting
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectRuleOfThirds, this.render));

    // Listen for changes to the bbox to update guide positioning
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectBbox, this.render));
  }

  initialize = () => {
    this.log.debug('Initializing composition guide module');
    this.render();
  };

  /**
   * Renders the composition guide. The guide is only visible when the setting is enabled.
   */
  render = () => {
    const ruleOfThirds = this.manager.stateApi.getSettings().ruleOfThirds;
    const bbox = this.manager.stateApi.runSelector(selectBbox);
    if (!bbox) {
      this.konva.group.visible(false);
      return;
    }
    const { x, y, width, height } = bbox.rect;

    this.konva.group.visible(ruleOfThirds);

    if (!ruleOfThirds) {
      return;
    }

    // Calculate the thirds positions of the bounding box
    const oneThirdX = x + width / 3;
    const twoThirdsX = x + (2 * width) / 3;
    const oneThirdY = y + height / 3;
    const twoThirdsY = y + (2 * height) / 3;

    // Update the vertical lines (divide the bbox into thirds vertically)
    this.konva.verticalLine1.points([oneThirdX, y, oneThirdX, y + height]);
    this.konva.verticalLine2.points([twoThirdsX, y, twoThirdsX, y + height]);

    // Update the horizontal lines (divide the bbox into thirds horizontally)
    this.konva.horizontalLine1.points([x, oneThirdY, x + width, oneThirdY]);
    this.konva.horizontalLine2.points([x, twoThirdsY, x + width, twoThirdsY]);
  };

  destroy = () => {
    this.log.debug('Destroying composition guide module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.group.destroy();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      visible: this.konva.group.visible(),
    };
  };
}
