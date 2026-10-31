import type { Project } from '@workbench/projectContracts';
import type { ScopedDefaults } from '@workbench/settings/contracts';
import type { WidgetInstanceContract, WidgetStateContract, WidgetTypeId } from '@workbench/widgetContracts';

import { getLayeredSettings } from '@workbench/settings/projectSettings';

const emptyWidgetState = (widgetId: WidgetTypeId): WidgetStateContract => ({
  id: widgetId,
  label: widgetId,
  values: {},
  version: 1,
});

/** Shared by every read of a missing instance; never written. */
const EMPTY_WIDGET_VALUES: Record<string, unknown> = {};

/** The widget instances alone: readers of a widget's values need nothing else from the project. */
export type WidgetInstanceHolder = Pick<Project, 'widgetInstances'>;

export const getProjectWidgetInstance = (
  project: WidgetInstanceHolder,
  widgetId: WidgetTypeId
): WidgetInstanceContract | undefined =>
  project.widgetInstances[widgetId] ??
  Object.values(project.widgetInstances).find((instance) => instance.typeId === widgetId);

export const getProjectWidgetState = (project: WidgetInstanceHolder, widgetId: WidgetTypeId): WidgetStateContract =>
  getProjectWidgetInstance(project, widgetId)?.state ?? emptyWidgetState(widgetId);

/**
 * The widget's own values: the project's pins alone. Write paths and persistence read these; a display or runtime
 * read of a settings owner's values uses {@link getResolvedProjectWidgetValues}.
 */
export const getProjectWidgetValues = (
  project: WidgetInstanceHolder,
  widgetId: WidgetTypeId
): Record<string, unknown> => getProjectWidgetState(project, widgetId).values;

/** Per defaults record, per pins object: the merge, so repeated reads of unchanged inputs keep their identity. */
// Created on first use: a module-scope `new WeakMap()` reads a global, which makes the module side-effectful to
// Rolldown and keeps it (and every barrel re-exporting through it) out of tree-shaking.
let resolvedValuesCache: WeakMap<object, WeakMap<object, Record<string, unknown>>> | undefined;

/**
 * `values` over the user's defaults for widget type `typeId`, taking from the defaults only the type's layerable keys
 * (`getLayeredSettings`): a stray key in synced preferences never reaches a widget. Without such a default this is
 * `values` itself, so selectors keep their identity; with them the merge is memoized on both objects.
 */
export const applyWidgetDefaults = (
  values: Record<string, unknown>,
  typeId: WidgetTypeId,
  defaults: ScopedDefaults
): Record<string, unknown> => {
  const typeDefaults = defaults[typeId];
  const layered = typeDefaults ? getLayeredSettings().widgetValues[typeId] : undefined;
  if (!typeDefaults || !layered || !Object.keys(typeDefaults).some((key) => Object.hasOwn(layered, key))) {
    return values;
  }
  resolvedValuesCache ??= new WeakMap();
  let byValues = resolvedValuesCache.get(typeDefaults);
  if (!byValues) {
    byValues = new WeakMap();
    resolvedValuesCache.set(typeDefaults, byValues);
  }
  let resolved = byValues.get(values);
  if (!resolved) {
    resolved = { ...values };
    for (const key of Object.keys(typeDefaults)) {
      if (Object.hasOwn(layered, key) && !Object.hasOwn(values, key)) {
        resolved[key] = typeDefaults[key];
      }
    }
    byValues.set(values, resolved);
  }
  return resolved;
};

/** The values an instance reads with: its pins over the user's defaults for its type ({@link applyWidgetDefaults}). */
export const resolveWidgetValues = (
  instance: WidgetInstanceContract | undefined,
  defaults: ScopedDefaults
): Record<string, unknown> =>
  instance ? applyWidgetDefaults(instance.state.values, instance.typeId, defaults) : EMPTY_WIDGET_VALUES;

/**
 * {@link getProjectWidgetValues} with the user's defaults applied under the project's pins; a project without the
 * widget reads the defaults alone, as a new instance would.
 */
export const getResolvedProjectWidgetValues = (
  project: WidgetInstanceHolder,
  widgetId: WidgetTypeId,
  defaults: ScopedDefaults
): Record<string, unknown> =>
  applyWidgetDefaults(
    getProjectWidgetInstance(project, widgetId)?.state.values ?? EMPTY_WIDGET_VALUES,
    widgetId,
    defaults
  );
