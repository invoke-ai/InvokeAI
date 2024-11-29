import { $focusedRegion } from 'common/hooks/focus';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Coordinate } from 'features/controlLayers/store/types';
import type { Logger } from 'roarr';

type CanvasMoveToolModuleConfig = {
  /**
   * The number of pixels to nudge the entity by when moving with the arrow keys.
   */
  NUDGE_PX: number;
};

const DEFAULT_CONFIG: CanvasMoveToolModuleConfig = {
  NUDGE_PX: 1,
};

type NudgeKey = 'ArrowLeft' | 'ArrowRight' | 'ArrowUp' | 'ArrowDown';

export class CanvasMoveToolModule extends CanvasModuleBase {
  readonly type = 'move_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasMoveToolModuleConfig = DEFAULT_CONFIG;
  nudgeOffsets: Record<NudgeKey, Coordinate>;

  constructor(parent: CanvasToolModule) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);
    this.log.debug('Creating module');

    this.nudgeOffsets = {
      ArrowLeft: { x: -this.config.NUDGE_PX, y: 0 },
      ArrowRight: { x: this.config.NUDGE_PX, y: 0 },
      ArrowUp: { x: 0, y: -this.config.NUDGE_PX },
      ArrowDown: { x: 0, y: this.config.NUDGE_PX },
    };
  }

  isNudgeKey(key: string): key is NudgeKey {
    return this.nudgeOffsets[key as NudgeKey] !== undefined;
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

  nudge = (nudgeKey: NudgeKey) => {
    if ($focusedRegion.get() !== 'canvas') {
      return;
    }

    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!selectedEntity) {
      return;
    }

    if (
      selectedEntity.$isDisabled.get() ||
      selectedEntity.$isEmpty.get() ||
      selectedEntity.$isLocked.get() ||
      selectedEntity.$isEntityTypeHidden.get()
    ) {
      return;
    }

    const isBusy = this.manager.$isBusy.get();
    const isMoveToolSelected = this.parent.$tool.get() === 'move';
    const isThisEntityTransforming = this.manager.stateApi.$transformingAdapter.get() === selectedEntity;

    if (isBusy) {
      // When the canvas is busy, we shouldn't allow nudging - except when the canvas is busy transforming the selected
      // entity. Nudging is allowed during transformation, regardless of the selected tool.
      if (!isThisEntityTransforming) {
        return;
      }
    } else {
      // Otherwise, the canvas is not busy, and we should only allow nudging when the move tool is selected.
      if (!isMoveToolSelected) {
        return;
      }
    }

    const offset = this.nudgeOffsets[nudgeKey];
    selectedEntity.transformer.nudgeBy(offset);
  };
}
