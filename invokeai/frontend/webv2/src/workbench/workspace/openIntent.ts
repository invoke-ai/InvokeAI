import type { LayoutPresetId, WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

/** What the user wants to see. Never where. */
export interface OpenIntent {
  typeId: WidgetTypeId;
  /** 'stage': it must take the center. 'panel': a side panel, the status bar's panel or a window. Omit when either does. */
  surface?: 'stage' | 'panel';
}

export type OpenResult =
  | { ok: true; workspaceId: LayoutPresetId; changedWorkspace: boolean; region?: WidgetRegion }
  | { ok: false; reason: 'unavailable' | 'not-placed' };
