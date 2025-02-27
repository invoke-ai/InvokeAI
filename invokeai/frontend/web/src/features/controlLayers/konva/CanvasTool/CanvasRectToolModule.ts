import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { floorCoord, getPrefixedId, offsetCoord } from 'features/controlLayers/konva/util';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Logger } from 'roarr';

export class CanvasRectToolModule extends CanvasModuleBase {
  readonly type = 'rect_tool';
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
    this.manager.stage.setCursor('crosshair');
  };

  onStagePointerDown = async (_e: KonvaEventObject<PointerEvent>) => {
    const cursorPos = this.parent.$cursorPos.get();
    const isPrimaryPointerDown = this.parent.$isPrimaryPointerDown.get();
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!cursorPos || !isPrimaryPointerDown || !selectedEntity) {
      /**
       * Can't do anything without:
       * - A cursor position: the cursor is not on the stage
       * - The mouse is down: the user is not drawing
       * - A selected entity: there is no entity to draw on
       */
      return;
    }

    const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);

    await selectedEntity.bufferRenderer.setBuffer({
      id: getPrefixedId('rect'),
      type: 'rect',
      rect: { x: Math.round(normalizedPoint.x), y: Math.round(normalizedPoint.y), width: 0, height: 0 },
      color: this.manager.stateApi.getCurrentColor(),
    });
  };

  onStagePointerUp = (_e: KonvaEventObject<PointerEvent>) => {
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    if (!selectedEntity) {
      return;
    }

    if (selectedEntity.bufferRenderer.state?.type === 'rect' && selectedEntity.bufferRenderer.hasBuffer()) {
      selectedEntity.bufferRenderer.commitBuffer();
    } else {
      selectedEntity.bufferRenderer.clearBuffer();
    }
  };

  onStagePointerMove = async (_e: KonvaEventObject<PointerEvent>) => {
    const cursorPos = this.parent.$cursorPos.get();

    if (!cursorPos) {
      return;
    }

    if (!this.parent.$isPrimaryPointerDown.get()) {
      return;
    }

    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!selectedEntity) {
      return;
    }

    const bufferState = selectedEntity.bufferRenderer.state;

    if (!bufferState) {
      return;
    }

    if (bufferState.type !== 'rect') {
      return;
    }

    const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);
    const alignedPoint = floorCoord(normalizedPoint);
    bufferState.rect.width = Math.round(alignedPoint.x - bufferState.rect.x);
    bufferState.rect.height = Math.round(alignedPoint.y - bufferState.rect.y);
    await selectedEntity.bufferRenderer.setBuffer(bufferState);
  };
}
