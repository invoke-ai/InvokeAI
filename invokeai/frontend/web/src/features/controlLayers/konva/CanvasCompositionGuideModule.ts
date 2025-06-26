import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectRuleOfFourGuide } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectBbox } from 'features/controlLayers/store/selectors';
import Konva from 'konva';
import type { Logger } from 'roarr';

/**
 * Renders the rule of 4 composition guide overlay on the canvas.
 * The guide shows a 2x2 grid within the bounding box to help with composition.
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
   * - Individual line objects for the grid
   */
  konva: {
    group: Konva.Group;
    verticalLine: Konva.Line;
    horizontalLine: Konva.Line;
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
      verticalLine: new Konva.Line({
        name: `${this.type}:vertical_line`,
        listening: false,
        stroke: 'rgba(255, 255, 255, 0.8)',
        strokeWidth: 1,
        strokeScaleEnabled: false,
        perfectDrawEnabled: false,
        dash: [5, 5],
      }),
      horizontalLine: new Konva.Line({
        name: `${this.type}:horizontal_line`,
        listening: false,
        stroke: 'rgba(255, 255, 255, 0.8)',
        strokeWidth: 1,
        strokeScaleEnabled: false,
        perfectDrawEnabled: false,
        dash: [5, 5],
      }),
    };

    this.konva.group.add(this.konva.verticalLine);
    this.konva.group.add(this.konva.horizontalLine);

    // Listen for changes to the rule of 4 guide setting
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectRuleOfFourGuide, this.render));

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
    const ruleOfFourGuide = this.manager.stateApi.getSettings().ruleOfFourGuide;
    const { x, y, width, height } = this.manager.stateApi.runSelector(selectBbox).rect;

    this.konva.group.visible(ruleOfFourGuide);

    if (!ruleOfFourGuide) {
      return;
    }

    // Calculate the center point of the bounding box
    const centerX = x + width / 2;
    const centerY = y + height / 2;

    // Update the vertical line (divides the bbox into left and right halves)
    this.konva.verticalLine.points([centerX, y, centerX, y + height]);

    // Update the horizontal line (divides the bbox into top and bottom halves)
    this.konva.horizontalLine.points([x, centerY, x + width, centerY]);
  };

  destroy = () => {
    this.log.debug('Destroying composition guide module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.group.destroy();
  };
}