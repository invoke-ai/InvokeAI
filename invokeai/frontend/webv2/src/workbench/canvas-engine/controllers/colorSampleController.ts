import type { ColorSamplePhase } from '@workbench/canvas-engine/capabilities';
import type { ToolId } from '@workbench/canvas-engine/types';

export interface ColorSampleControllerOptions {
  /** The tool the user is on; a settled request returns to it. */
  readonly getActiveToolId: () => ToolId;
  /** Switches tools directly; the controller settles its own request around the switch. */
  readonly setTool: (toolId: ToolId) => void;
  /** Ends a held temporary tool, so a request records the tool the user was really on. */
  readonly endHeldTool: () => void;
}

type ColorSampleRouter = (hex: string, phase: ColorSamplePhase) => boolean;

/**
 * Routes eyedropper picks. A one-shot request takes a pick's previews and settles on its release commit, after
 * structural edits are allowed; Escape, a tool change and disposal settle it with null. Unclaimed picks go to the
 * workbench router while one is installed. A pick belongs to the sink that took its first preview: its commit or
 * cancel goes there alone, so a request armed or settled mid-pick never strays into the router, nor the reverse.
 */
export class ColorSampleController {
  private request: { previousToolId: ToolId; resolve: (hex: string | null) => void } | null = null;
  private router: ColorSampleRouter | null = null;
  // Which sink took the current pick's first preview; null between picks.
  private pickSink: 'request' | 'router' | null = null;
  private disposed = false;

  constructor(private readonly deps: ColorSampleControllerOptions) {}

  /** Arms the color picker and resolves the picked hex on release, or null when cancelled. */
  requestSample(): Promise<string | null> {
    // A second request supersedes the first; the earlier caller gets a cancel.
    this.settle(null, false);
    if (this.disposed) {
      return Promise.resolve(null);
    }
    this.deps.endHeldTool();
    return new Promise<string | null>((resolve) => {
      this.request = { previousToolId: this.deps.getActiveToolId(), resolve };
      this.deps.setTool('colorPicker');
    });
  }

  /** Takes one sample of a pick; returns whether a sink consumed it, else the picker falls back to the brush. */
  route(hex: string, phase: ColorSamplePhase): boolean {
    if (phase === 'preview') {
      this.pickSink ??= this.request ? 'request' : 'router';
    }
    const sink = this.pickSink;
    if (phase !== 'preview') {
      this.pickSink = null;
    }
    if (sink === 'request') {
      if (phase === 'commit') {
        this.settle(hex, true);
      }
      return true;
    }
    return sink === 'router' ? (this.router?.(hex, phase) ?? false) : false;
  }

  /** Installs the workbench router; the returned cleanup removes it only while it is still the installed one. */
  setRouter(router: ColorSampleRouter): () => void {
    this.router = router;
    return () => {
      if (this.router === router) {
        this.router = null;
      }
    };
  }

  /** Any move off the eyedropper abandons the sample the caller is awaiting; the requested switch wins. */
  onToolChange(toolId: ToolId): void {
    if (toolId !== 'colorPicker') {
      this.settle(null, false);
    }
  }

  /** An armed request is the most recent thing the user opted into, so Escape takes it back first. */
  onEscape(): boolean {
    if (!this.request) {
      return false;
    }
    this.settle(null, true);
    return true;
  }

  private settle(hex: string | null, restoreTool: boolean): void {
    const request = this.request;
    if (!request) {
      return;
    }
    this.request = null;
    request.resolve(hex);
    if (restoreTool) {
      this.deps.setTool(request.previousToolId);
    }
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.settle(null, false);
    this.router = null;
    this.pickSink = null;
  }
}
