import type { CanvasToolCapability } from '@workbench/canvas-engine/api';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { ToolId } from '@workbench/canvas-engine/types';

export interface InteractionControllerOptions {
  readonly initialToolId?: ToolId;
  readonly getTool: (toolId: ToolId) => Tool | undefined;
  readonly getToolContext: () => ToolContext;
  readonly isLocked: () => boolean;
  readonly beforeSwitch?: (from: ToolId, to: ToolId, options?: { temporary?: boolean }) => void;
  readonly publishActiveTool: (toolId: ToolId) => void;
  readonly updateCursor: () => void;
  readonly invalidateOverlay: () => void;
  readonly stepBrushSize: (direction: 1 | -1) => void;
}

/** Owns active-tool transitions and the public tool command boundary. */
export class InteractionController {
  readonly tools: CanvasToolCapability;
  private activeToolId: ToolId;
  private disposed = false;

  constructor(private readonly options: InteractionControllerOptions) {
    this.activeToolId = options.initialToolId ?? 'view';
    this.tools = {
      setTool: (toolId, switchOptions) => this.setTool(toolId, switchOptions),
      stepBrushSize: (direction) => {
        if (!this.disposed) {
          options.stepBrushSize(direction);
        }
      },
    };
  }

  getActiveToolId(): ToolId {
    return this.activeToolId;
  }

  getActiveTool(): Tool | undefined {
    return this.options.getTool(this.activeToolId);
  }

  setTool(toolId: ToolId, switchOptions?: { temporary?: boolean }): void {
    if (this.disposed || (this.options.isLocked() && toolId !== 'view') || toolId === this.activeToolId) {
      return;
    }
    const previous = this.activeToolId;
    this.options.beforeSwitch?.(previous, toolId, switchOptions);
    this.options.getTool(previous)?.onDeactivate?.(this.options.getToolContext(), switchOptions);
    this.activeToolId = toolId;
    this.options.getTool(toolId)?.onActivate?.(this.options.getToolContext(), switchOptions);
    this.options.publishActiveTool(toolId);
    this.options.updateCursor();
    this.options.invalidateOverlay();
  }

  dispose(): void {
    this.disposed = true;
  }
}
