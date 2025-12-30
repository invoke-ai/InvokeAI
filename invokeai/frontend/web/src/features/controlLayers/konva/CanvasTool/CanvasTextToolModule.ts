import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { type CanvasTextSettingsState, selectCanvasTextSlice } from 'features/controlLayers/store/canvasTextSlice';
import type { CanvasImageState, Coordinate, RgbaColor, Tool } from 'features/controlLayers/store/types';
import { getFontStackById, TEXT_RASTER_PADDING } from 'features/controlLayers/text/textConstants';
import {
  buildFontDescriptor,
  calculateLayerPosition,
  hasVisibleGlyphs,
  measureTextContent,
  renderTextToCanvas,
  type TextMeasureConfig,
} from 'features/controlLayers/text/textRenderer';
import { type TextSessionStatus, transitionTextSessionStatus } from 'features/controlLayers/text/textSessionMachine';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';

type CanvasTextSessionState = {
  id: string;
  anchor: Coordinate;
  position: Coordinate | null;
  status: CanvasTextSessionStatus;
  createdAt: number;
  text: string;
};

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
    label: Konva.Text;
  };

  $session = atom<CanvasTextSessionState | null>(null);
  private subscriptions = new Set<() => void>();
  private cursorHeight = 0;
  private cursorMetricsKey: string | null = null;

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
      label: new Konva.Text({
        name: `${this.type}:label`,
        text: 'T',
        listening: false,
        perfectDrawEnabled: false,
      }),
    };

    this.konva.group.add(this.konva.cursor);
    this.konva.group.add(this.konva.label);
    this.konva.label.visible(true);

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
  };

  private setCursorDimensions = (settings: CanvasTextSettingsState) => {
    const onePixel = this.manager.stage.unscale(this.config.CURSOR_MIN_WIDTH_PX);
    const cursorWidth = Math.max(onePixel * 2, onePixel);
    const metricsKey = `${settings.fontId}|${settings.fontSize}|${settings.bold}|${settings.italic}|${settings.lineHeight}`;
    if (this.cursorMetricsKey !== metricsKey) {
      const measureConfig: TextMeasureConfig = {
        text: 'Mg',
        fontSize: settings.fontSize,
        fontFamily: getFontStackById(settings.fontId),
        fontWeight: settings.bold ? 700 : 400,
        fontStyle: settings.italic ? 'italic' : 'normal',
        lineHeight: settings.lineHeight,
      };
      const metrics = measureTextContent(measureConfig);
      this.cursorHeight = Math.max(metrics.lineHeightPx, settings.fontSize) + TEXT_RASTER_PADDING * 2;
      this.cursorMetricsKey = metricsKey;
    }
    const height = this.cursorHeight || settings.fontSize + TEXT_RASTER_PADDING * 2;
    this.konva.cursor.setAttrs({
      width: cursorWidth,
      height,
    });
    this.konva.label.setAttrs({
      fontFamily: getFontStackById('uiSerif'),
      fontSize: Math.max(12, height * 0.35),
      fontStyle: settings.bold ? '700' : '400',
      fill: 'rgba(0, 0, 0, 1)',
      stroke: 'rgba(255, 255, 255, 1)',
      strokeWidth: Math.max(1, onePixel),
    });
    this.konva.cursor.fill('rgba(0, 0, 0, 1)');
    this.konva.cursor.stroke('rgba(255, 255, 255, 1)');
    this.konva.cursor.strokeWidth(onePixel);
  };

  private setCursorPosition = (cursor: Coordinate, _settings: CanvasTextSettingsState) => {
    const top = cursor.y - TEXT_RASTER_PADDING;
    this.konva.cursor.setAttrs({
      x: cursor.x,
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
    this.konva.label.visible(visible);
  };

  onStagePointerMove = (e: KonvaEventObject<PointerEvent>) => {
    if (e.target !== this.parent.konva.stage) {
      return;
    }
    const cursorPos = this.parent.$cursorPos.get();
    if (!cursorPos) {
      return;
    }
  };

  onStagePointerEnter = (e: KonvaEventObject<PointerEvent>) => {
    if (e.target !== this.parent.konva.stage) {
      return;
    }
    const cursorPos = this.parent.$cursorPos.get();
    if (!cursorPos) {
      return;
    }
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
    const color = canvasSettings.activeColor === 'fgColor' ? canvasSettings.fgColor : canvasSettings.bgColor;

    this.$session.set({
      ...session,
      status: coerceSessionStatus(transitionTextSessionStatus(session.status, 'COMMIT')),
    });

    void this.commitSession(session, rawText, textSettings, color);
  };

  private commitSession = async (
    session: CanvasTextSessionState,
    rawText: string,
    textSettings: CanvasTextSettingsState,
    color: RgbaColor
  ) => {
    if (typeof document !== 'undefined' && document.fonts?.load) {
      const fontSpec = buildFontDescriptor({
        fontFamily: getFontStackById(textSettings.fontId),
        fontWeight: textSettings.bold ? 700 : 400,
        fontStyle: textSettings.italic ? 'italic' : 'normal',
        fontSize: textSettings.fontSize,
      });
      try {
        await document.fonts.load(fontSpec);
        await document.fonts.ready;
      } catch {
        // Ignore font load failures and proceed with available metrics.
      }
    }

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
    const position = session.position ? { x: session.position.x, y: session.position.y } : fallbackPosition;

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

  private buildLayerName = (text: string) => {
    const flattened = text.replace(/\s+/g, ' ').trim();
    if (!flattened) {
      return 'Text';
    }
    return flattened.length > 32 ? `${flattened.slice(0, 29)}…` : flattened;
  };
}
