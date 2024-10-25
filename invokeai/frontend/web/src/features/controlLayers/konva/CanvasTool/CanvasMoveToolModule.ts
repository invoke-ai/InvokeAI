import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Logger } from 'roarr';

// Typo insurance
const KEY_LEFT = 'ArrowLeft';
const KEY_RIGHT = 'ArrowRight';
const KEY_UP = 'ArrowUp';
const KEY_DOWN = 'ArrowDown';

export class CanvasMoveToolModule extends CanvasModuleBase {
  readonly type = 'move_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  constructor(parent: CanvasToolModule) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');
  }

  syncCursorStyle = () => {
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    if (!selectedEntity) {
      this.manager.stage.setCursor('not-allowed');
    } else {
      // The cursor is on an entity, defer to transformer to handle the cursor
      selectedEntity.transformer.syncCursorStyle();
    }
  };

  onKeyDown = (e: KeyboardEvent) => {
    // Support moving via arrow keys
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    let offset_x;
    let offset_y;
    switch (e.key) {
      case KEY_LEFT:
        offset_x = -1;
        break;
      case KEY_RIGHT:
        offset_x = 1;
        break;
      case KEY_UP:
        offset_y = -1;
        break;
      case KEY_DOWN:
        offset_y = 1;
        break;
    }
    if (offset_x !== undefined) {
      selectedEntity?.konva.layer.x(selectedEntity?.konva.layer.x() + offset_x);
    }
    if (offset_y !== undefined) {
      selectedEntity?.konva.layer.y(selectedEntity?.konva.layer.y() + offset_y);
    }
  };
}
