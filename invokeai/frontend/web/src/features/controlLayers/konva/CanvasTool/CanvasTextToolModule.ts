import { throttle } from 'es-toolkit/compat';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { getColorAtCoordinate, getPrefixedId } from 'features/controlLayers/konva/util';
import {
  type CanvasTextSettingsState,
  selectCanvasTextSlice,
} from 'features/controlLayers/store/canvasTextSlice';
import type { CanvasImageState, Coordinate, RgbaColor, Tool } from 'features/controlLayers/store/types';
import { RGBA_BLACK, RGBA_WHITE } from 'features/controlLayers/store/types';
import { getFontStackById, TEXT_RASTER_PADDING } from 'features/controlLayers/text/textConstants';
import { calculateLayerPosition, hasVisibleGlyphs, renderTextToCanvas } from 'features/controlLayers/text/textRenderer';
import { type TextSessionStatus, transitionTextSessionStatus } from 'features/controlLayers/text/textSessionMachine';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';

export type CanvasTextSessionState = {
  id: string;
  anchor: Coordinate;
  position: Coordinate | null;
  status: CanvasTextSessionStatus;
  createdAt: number;
  text: string;
};

type CursorColor = RgbaColor;

type CanvasTextToolModuleConfig = {
  CURSOR_MIN_WIDTH_PX: number;
};

type CanvasTextSessionStatus = Exclude<TextSessionStatus, 'idle'>;

const coerceSessionStatus = (status: TextSessionStatus): CanvasTextSessionStatus => {
  if (status === 'idle') {
    return 'pending';
  }
  return status;
};

const DEFAULT_CONFIG: CanvasTextToolModuleConfig = {
  CURSOR_MIN_WIDTH_PX: 1.5,
};

export class CanvasTextToolModule extends CanvasModuleBase {
  readonly type = 'text_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasTextToolModuleConfig = DEFAULT_CONFIG;

  konva: {
    group: Konva.Group;
    cursor: Konva.Rect;
    cursorOutline: Konva.Rect;
    label: Konva.Text;
  };

  $session = atom<CanvasTextSessionState | null>(null);
  $cursorColor = atom<CursorColor>(RGBA_WHITE);
  private lastStageCursor: Coordinate | null = null;
  private subscriptions = new Set<() => void>();

  constructor(parent: CanvasToolModule) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      cursor: new Konva.Rect({
        name: `${this.type}:cursor`,
        width: 1,
        height: 10,
        listening: false,
        perfectDrawEnabled: false,
      }),
      cursorOutline: new Konva.Rect({
        name: `${this.type}:cursorOutline`,
        width: 1,
        height: 10,
        listening: false,
        perfectDrawEnabled: false,
      }),
      label: new Konva.Text({
        name: `${this.type}:label`,
        text: 'T',
        listening: false,
        perfectDrawEnabled: false,
      }),
    };

    this.konva.group.add(this.konva.cursorOutline);
    this.konva.group.add(this.konva.cursor);
    this.konva.group.add(this.konva.label);
    this.konva.label.visible(false);

    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectCanvasTextSlice, () => {
        this.render();
      })
    );
    this.subscriptions.add(
      this.parent.$cursorPos.listen(() => {
        this.render();
      })
    );
  }

  destroy = () => {
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.group.destroy();
  };

  syncCursorStyle = () => {
    this.manager.stage.setCursor('none');
  };

  render = () => {
    const textSettings = this.manager.stateApi.runSelector(selectCanvasTextSlice);
    const cursorPos = this.parent.$cursorPos.get();
    const session = this.$session.get();

    if (this.parent.$tool.get() !== 'text' || !cursorPos || (session && session.status === 'editing')) {
      this.setVisibility(false);
      return;
    }

    this.setVisibility(true);
    this.setCursorDimensions(textSettings);
    this.setCursorPosition(cursorPos.relative, textSettings);
    if (this.lastStageCursor) {
      this.sampleCursorColor(this.lastStageCursor, true);
    }
  };

  private setCursorDimensions = (settings: CanvasTextSettingsState) => {
    const onePixel = this.manager.stage.unscale(this.config.CURSOR_MIN_WIDTH_PX);
    const height = settings.fontSize + TEXT_RASTER_PADDING * 2;
    this.konva.cursor.setAttrs({
      width: onePixel,
      height,
    });
    this.konva.cursorOutline.setAttrs({
      width: onePixel * 3,
      height,
    });
    this.konva.label.setAttrs({
      fontSize: Math.max(12, height * 0.35),
      fontStyle: settings.bold ? '700' : '400',
    });
    const color = this.$cursorColor.get();
    const fill = `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a})`;
    const luminance = (0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b) / 255;
    const inverseColor = luminance > 0.5 ? RGBA_BLACK : RGBA_WHITE;
    const inverse = `rgba(${inverseColor.r}, ${inverseColor.g}, ${inverseColor.b}, ${inverseColor.a})`;
    this.konva.cursor.fill(fill);
    this.konva.cursorOutline.stroke(inverse);
    this.konva.cursorOutline.strokeWidth(onePixel);
    this.konva.label.fill(fill);
  };

  private setCursorPosition = (cursor: Coordinate, _settings: CanvasTextSettingsState) => {
    const top = cursor.y - TEXT_RASTER_PADDING;
    this.konva.cursor.setAttrs({
      x: cursor.x,
      y: top,
    });
    this.konva.cursorOutline.setAttrs({
      x: cursor.x - this.konva.cursorOutline.width() / 2 + this.konva.cursor.width() / 2,
      y: top,
    });
    const labelFontSize = this.konva.label.fontSize();
    this.konva.label.setAttrs({
      x: cursor.x + this.konva.cursor.width() * 1.5,
      y: top + this.konva.cursor.height() - labelFontSize * 0.6,
    });
  };

  private setVisibility = (visible: boolean) => {
    this.konva.group.visible(visible);
  };

  onStagePointerMove = (e: KonvaEventObject<PointerEvent>) => {
    if (e.target !== this.parent.konva.stage) {
      return;
    }
    const cursorPos = this.parent.$cursorPos.get();
    if (!cursorPos) {
      return;
    }
    this.lastStageCursor = cursorPos.absolute;
    this.sampleCursorColor(cursorPos.absolute);
  };

  onStagePointerEnter = (e: KonvaEventObject<PointerEvent>) => {
    if (e.target !== this.parent.konva.stage) {
      return;
    }
    const cursorPos = this.parent.$cursorPos.get();
    if (!cursorPos) {
      return;
    }
    this.lastStageCursor = cursorPos.absolute;
    this.sampleCursorColor(cursorPos.absolute, true);
  };

  onStagePointerDown = (e: KonvaEventObject<PointerEvent>) => {
    // Only allow left-click/primary pointer to begin typing sessions.
    if (e.target !== this.parent.konva.stage || e.evt.button !== 0) {
      return;
    }
    const cursorPos = this.parent.$cursorPos.get();
    if (!cursorPos) {
      return;
    }
    if (this.$session.get()) {
      this.commitExistingSession();
    }
    this.beginSession(cursorPos.relative);
  };

  beginSession = (anchor: Coordinate) => {
    const current = this.$session.get();
    if (current && current.status === 'editing') {
      return;
    }
    const id = getPrefixedId('text_session');
    const status = coerceSessionStatus(transitionTextSessionStatus('idle', 'BEGIN'));
    this.$session.set({
      id,
      anchor,
      position: null,
      status,
      createdAt: Date.now(),
      text: '',
    });
  };

  markSessionEditing = (id: string) => {
    const current = this.$session.get();
    if (!current || current.id !== id) {
      return;
    }
    const nextStatus = coerceSessionStatus(transitionTextSessionStatus(current.status, 'EDIT'));
    this.$session.set({
      ...current,
      status: nextStatus,
    });
  };

  clearSession = () => {
    this.$session.set(null);
  };

  updateSessionText = (sessionId: string, text: string) => {
    const current = this.$session.get();
    if (!current || current.id !== sessionId) {
      return;
    }
    this.$session.set({ ...current, text });
  };

  updateSessionPosition = (sessionId: string, position: Coordinate) => {
    const current = this.$session.get();
    if (!current || current.id !== sessionId) {
      return;
    }
    this.$session.set({ ...current, position });
  };

  commitExistingSession = () => {
    const session = this.$session.get();
    if (!session) {
      return;
    }
    this.requestCommit(session.id);
  };

  onToolChanged = (prevTool: Tool, nextTool: Tool) => {
    if (prevTool === 'text' && nextTool !== 'text') {
      this.commitExistingSession();
    }
  };

  requestCommit = (sessionId: string) => {
    const session = this.$session.get();
    if (!session || session.id !== sessionId) {
      return;
    }
    const rawText = session.text.replace(/\r/g, '');
    if (!hasVisibleGlyphs(rawText)) {
      this.clearSession();
      return;
    }

    const textSettings = this.manager.stateApi.runSelector(selectCanvasTextSlice);
    const canvasSettings = this.manager.stateApi.getSettings();
    const color =
      canvasSettings.activeColor === 'fgColor' ? canvasSettings.fgColor : canvasSettings.bgColor;

    this.$session.set({
      ...session,
      status: coerceSessionStatus(transitionTextSessionStatus(session.status, 'COMMIT')),
    });

    const renderResult = renderTextToCanvas({
      text: rawText,
      fontSize: textSettings.fontSize,
      fontFamily: getFontStackById(textSettings.fontId),
      fontWeight: textSettings.bold ? 700 : 400,
      fontStyle: textSettings.italic ? 'italic' : 'normal',
      underline: textSettings.underline,
      strikethrough: textSettings.strikethrough,
      lineHeight: textSettings.lineHeight,
      color,
      alignment: textSettings.alignment,
      padding: TEXT_RASTER_PADDING,
      devicePixelRatio: window.devicePixelRatio ?? 1,
    });

    const dataURL = renderResult.canvas.toDataURL('image/png');
    const imageState: CanvasImageState = {
      id: getPrefixedId('image'),
      type: 'image',
      image: {
        dataURL,
        width: renderResult.totalWidth,
        height: renderResult.totalHeight,
      },
    };

    const fallbackPosition = calculateLayerPosition(
      session.anchor,
      textSettings.alignment,
      renderResult.contentWidth,
      TEXT_RASTER_PADDING
    );
    const position = session.position
      ? { x: Math.round(session.position.x), y: Math.round(session.position.y) }
      : fallbackPosition;

    const selectedAdapter = this.manager.stateApi.getSelectedEntityAdapter();
    const addAfter =
      selectedAdapter && selectedAdapter.state.type === 'raster_layer' ? selectedAdapter.state.id : undefined;

    this.manager.stateApi.addRasterLayer({
      overrides: {
        objects: [imageState],
        position,
        name: this.buildLayerName(rawText),
      },
      isSelected: true,
      addAfter,
    });

    this.clearSession();
  };

  private computeCursorContrast = (color: RgbaColor): CursorColor => {
    const luminance = (0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b) / 255;
    return luminance > 0.6 ? RGBA_BLACK : RGBA_WHITE;
  };

  private buildLayerName = (text: string) => {
    const flattened = text.replace(/\s+/g, ' ').trim();
    if (!flattened) {
      return 'Text';
    }
    return flattened.length > 32 ? `${flattened.slice(0, 29)}…` : flattened;
  };

  private sampleCursorColor = throttle((coordinate: Coordinate, force: boolean = false) => {
    if (!force && !this.parent.getCanDraw()) {
      return;
    }
    try {
      const color = getColorAtCoordinate(this.manager.stage.konva.stage, coordinate);
      if (!color) {
        this.$cursorColor.set(RGBA_WHITE);
        return;
      }
      const contrastColor = this.computeCursorContrast(color);
      this.$cursorColor.set(contrastColor);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.log.error('Failed to sample cursor color: %s', message);
      this.$cursorColor.set(RGBA_WHITE);
    }
  }, 100);
}
