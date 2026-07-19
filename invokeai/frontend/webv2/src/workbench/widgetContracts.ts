import type { TFunction } from 'i18next';
import type { ComponentType, ExoticComponent, JSXElementConstructor, SVGProps } from 'react';

import type { DeveloperLogNamespace } from './diagnostics/contracts';
import type { GraphId } from './graphContracts';
import type { InvocationSourceId } from './invocationContracts';
import type { WidgetRegion } from './layoutContracts';

export type FirstPartyWidgetTypeId =
  | 'autosave-status'
  | 'canvas'
  | 'diagnostics'
  | 'gallery'
  | 'generate'
  | 'history-controls'
  | 'layers'
  | 'notifications'
  | 'preview'
  | 'project'
  | 'queue'
  | 'server-status'
  | 'upscale'
  | 'users'
  | 'version-status'
  | 'workflow';

export type WidgetTypeId = FirstPartyWidgetTypeId | (string & {});

export type WidgetInstanceId = string;

export type WidgetId = WidgetTypeId;

export type WorkbenchRegion = 'left' | 'right' | 'center' | 'bottom' | 'dialog' | 'popover';

export interface WidgetStateContract {
  id: WidgetTypeId;
  label: string;
  version: 1;
  values: Record<string, unknown>;
  graphId?: GraphId;
}

export type WidgetStateMap = Record<string, WidgetStateContract>;

export interface WidgetInstanceContract {
  id: WidgetInstanceId;
  typeId: WidgetTypeId;
  title?: string;
  state: WidgetStateContract;
  createdAt: string;
}

export interface WidgetInstanceRuntimeMeta {
  id: WidgetInstanceId;
  typeId: WidgetTypeId;
  title?: string;
  createdAt: string;
}

export interface GraphBearingSurfaceContract {
  surfaceId: string;
  widgetId: WidgetId;
  label: string;
  sourceId: InvocationSourceId;
  graphId: GraphId;
  region: WorkbenchRegion;
  canSetSource: boolean;
  canPreviewGraph: boolean;
}

export interface WidgetViewProps {
  region: WorkbenchRegion;
  manifest: WidgetManifest;
  /** Chrome metadata only. Read reactive widget values through `runtime.state`. */
  instance: WidgetInstanceRuntimeMeta;
  runtime: WidgetRuntimeApi;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export type WidgetView = ComponentType<WidgetViewProps>;

export type WidgetHeaderActions = ComponentType<WidgetViewProps>;

export type WidgetHeaderMenu = ComponentType<WidgetViewProps>;

export type WidgetFooter = ComponentType<WidgetViewProps>;

export type WidgetHost = ComponentType;

/** Deferred render implementation shared by every slot for one widget type. */
export interface WidgetImplementation {
  view: WidgetView;
  host?: WidgetHost;
  headerActions?: WidgetHeaderActions;
  headerLabel?: WidgetHeaderLabel;
  headerMenu?: WidgetHeaderMenu;
  footer?: WidgetFooter;
}

export type WidgetImplementationLoadStatus = 'idle' | 'loading' | 'loaded' | 'failed';

/** Registry-owned, single-flight resource for a deferred widget implementation. */
export interface WidgetImplementationResource {
  getStatus: () => WidgetImplementationLoadStatus;
  load: () => Promise<WidgetImplementation>;
  preload: () => void;
  retry: () => Promise<WidgetImplementation>;
}

export interface WidgetLabelProps {
  region: WorkbenchRegion;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export type WidgetLabel = string | ((t: TFunction) => string);

export type WidgetHeaderLabel = ComponentType<WidgetLabelProps>;

export type WidgetIconComponent =
  | JSXElementConstructor<SVGProps<SVGSVGElement>>
  | ExoticComponent<SVGProps<SVGSVGElement>>;

export interface WidgetStateRegistration<State extends Record<string, unknown> = Record<string, unknown>> {
  version: 1;
  createInitial: () => State;
  migrate?: (state: unknown, fromVersion: number) => State;
  persistence?: 'project' | 'workspace' | 'session' | 'none';
}

export interface WidgetRuntimeApi<State extends Record<string, unknown> = Record<string, unknown>> {
  instanceId: WidgetInstanceId;
  typeId: WidgetTypeId;
  region: WorkbenchRegion;
  state: WidgetRuntimeStateApi<State>;
  diagnostics: WidgetDiagnosticsApi;
  commands: WidgetCommandApi;
  hotkeys: WidgetHotkeyApi;
  menus: WidgetMenuApi;
  palette: WidgetCommandPaletteApi;
  search: WidgetSearchApi;
  toolbars: WidgetToolbarApi;
  workbench: WidgetWorkbenchApi;
}

export interface WidgetDiagnosticsApi {
  logger: (namespace: DeveloperLogNamespace) => {
    debug: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    error: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    fatal: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    info: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    trace: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    warn: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
  };
}

export interface WidgetRuntimeStateApi<State extends Record<string, unknown> = Record<string, unknown>> {
  getSnapshot: () => Readonly<State>;
  patch: (values: Partial<State>) => void;
  set: (values: State) => void;
}

export interface WidgetContributionSource {
  instanceId: WidgetInstanceId;
  projectId: string;
  region: WorkbenchRegion;
  typeId: WidgetTypeId;
}

export interface WidgetCommandApi {
  execute: (commandId: string, ...args: unknown[]) => Promise<unknown>;
  executeForSource: (
    commandId: string,
    source: WidgetContributionSource | null,
    ...args: unknown[]
  ) => Promise<unknown>;
  register: (command: WidgetCommandContribution) => () => void;
}

export interface WidgetCommandContribution {
  id: string;
  title: string;
  handler: (...args: unknown[]) => unknown | Promise<unknown>;
  source?: WidgetContributionSource;
}

export interface WidgetHotkeyApi {
  register: (hotkey: WidgetHotkeyContribution) => () => void;
}

export interface WidgetHotkeyContribution {
  id: string;
  commandId: string;
  defaultKeys: string[];
  title: string;
  description?: string;
  scope?: 'focused-region' | 'global' | 'instance' | 'widget';
  source?: WidgetContributionSource;
  preventDefault?: boolean;
  allowInEditable?: boolean;
}

export interface WidgetMenuApi {
  register: (menu: WidgetMenuContribution) => () => void;
}

export interface WidgetMenuContribution {
  id: string;
  items: Array<{ commandId: string; group?: string }>;
  source?: WidgetContributionSource;
}

export interface WidgetCommandPaletteApi {
  register: (entry: WidgetCommandPaletteContribution) => () => void;
}

export interface WidgetCommandPaletteContribution {
  commandId: string;
  title: string;
  keywords?: string[];
}

export interface WidgetSearchApi {
  registerProvider: (provider: WidgetSearchProvider) => () => void;
}

export interface WidgetSearchProvider {
  id: string;
  label: string;
  search: (query: string) => Promise<WidgetSearchResult[]> | WidgetSearchResult[];
}

export interface WidgetSearchResult {
  id: string;
  title: string;
  subtitle?: string;
  commandId?: string;
}

export interface WidgetToolbarApi {
  register: (toolbar: WidgetToolbarContribution) => () => void;
}

export interface WidgetToolbarContribution {
  id: string;
  location: 'center.tabs.trailing' | 'status.left' | 'status.right';
  items: Array<{
    commandId: string;
    icon?: WidgetIconComponent;
    label?: string;
  }>;
}

export interface OpenWorkbenchWidgetOptions {
  createNew?: boolean;
  preferredRegions?: ReadonlyArray<WidgetRegion>;
  requireCenterView?: boolean;
}

export type WidgetWorkbenchApiResult =
  | { ok: true; region?: WidgetRegion }
  | { ok: false; reason: 'unavailable' | 'unsupported' | 'not-found' };

export interface WidgetWorkbenchApi {
  openWidget: (typeId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions) => WidgetWorkbenchApiResult;
  revealWidgetInstance: (instanceId: WidgetInstanceId) => WidgetWorkbenchApiResult;
  closeWidgetInstance: (instanceId: WidgetInstanceId) => WidgetWorkbenchApiResult;
}

/** Sections of the workbench settings dialog, addressable via `openWorkbenchSettings`. */
export type SettingsSectionId =
  | 'appearance'
  | 'behavior'
  | 'hotkeys'
  | 'project'
  | 'queue'
  | 'workflow'
  | 'developer'
  | 'workspace';

export interface WidgetManifest {
  /** Widget runtime API contract version. Defaults to 1 during registry normalization. */
  apiVersion?: 1;
  id: WidgetTypeId;
  label: WidgetLabel;
  headerLabel?: WidgetHeaderLabel;
  version: 1;
  allowedRegions: WidgetRegion[];
  allowMultiple: boolean;
  icon: WidgetIconComponent;
  bottomPanel?: 'expandable' | 'tooltip';
  centerPlacement?: 'toolbar' | 'view';
  /** Only offered while an admin is signed in to a multi-user backend. */
  requiresAdmin?: boolean;
  chrome?: {
    header?: 'hidden' | 'visible';
  };
  /**
   * Literal dynamic loader for all render slots. The registry owns caching,
   * failure state, and retry; callers never invoke this directly.
   */
  load: () => Promise<WidgetImplementation>;
  /** The deferred implementation contains a singleton host mounted at editor boot. */
  hasHost?: boolean;
  /** When set, the frame header shows a gear that opens this settings dialog section. */
  settingsSection?: SettingsSectionId;
  state?: WidgetStateRegistration;
  graphBearing?: {
    sourceId: InvocationSourceId;
    defaultGraphId: GraphId;
    surfaces: WorkbenchRegion[];
  };
  failurePolicy: {
    onRegistrationFailure: 'disable' | 'hide';
    isolateRenderFailure: boolean;
  };
}

export interface NormalizedWidgetManifest extends Omit<WidgetManifest, 'apiVersion' | 'state'> {
  apiVersion: 1;
  state: WidgetStateRegistration;
}

export interface RegisteredWidget {
  implementation: WidgetImplementationResource;
  manifest: NormalizedWidgetManifest;
  status: 'enabled' | 'disabled' | 'hidden';
  failure?: WidgetFailure;
}

export interface WidgetFailure {
  widgetId: WidgetTypeId;
  message: string;
  details: string;
  occurredAt: string;
}
