import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';

import { WIDGET_REGIONS } from '@workbench/layoutContracts';
import { defaultLayoutPreset } from '@workbench/layoutPresets';
import { canRenderWidgetInRegion } from '@workbench/widgetPlacementCommands';
import { getWidgetById, getWidgetsForRegion, isAdminWidgetSession } from '@workbench/widgetRegistry';

import type { LayoutAvailability } from './resolveLayout';
import type { RegionPresentation } from './resolveOpen';

import { getStageInstance, normalizeStructure } from './structure';

const presentationOf = (widget: RegisteredWidget | undefined, region: WidgetRegion): RegionPresentation => {
  if (widget?.status !== 'enabled') {
    return null;
  }

  return canRenderWidgetInRegion(widget, region) && !(region === 'bottom' && widget.manifest.bottomPanel === 'popover')
    ? 'view'
    : 'chip';
};

/** The resolver's region rules for this registry: enabled types the account may use, as a view or a chip. */
export const getRegionPresentation =
  (lookup: (region: WidgetRegion) => RegisteredWidget[]) =>
  (typeId: WidgetTypeId, region: WidgetRegion): RegionPresentation =>
    presentationOf(
      lookup(region).find((candidate) => candidate.manifest.id === typeId),
      region
    );

const NO_NEEDS: readonly WidgetTypeId[] = [];

/** The types a widget type acts on, from its manifest. */
export const getWidgetNeeds =
  (lookup: (region: WidgetRegion) => RegisteredWidget[]) =>
  (typeId: WidgetTypeId): readonly WidgetTypeId[] => {
    for (const region of WIDGET_REGIONS) {
      const widget = lookup(region).find((candidate) => candidate.manifest.id === typeId);

      if (widget) {
        return widget.manifest.needs ?? NO_NEEDS;
      }
    }

    return NO_NEEDS;
  };

/** The registry's answers for one session, read once: what each type shows as in each region. */
export const createRegistryLayoutAvailability = (): LayoutAvailability => {
  const presentations = new Map(
    WIDGET_REGIONS.map((region) => [
      region,
      new Map(getWidgetsForRegion(region).map((widget) => [widget.manifest.id, presentationOf(widget, region)])),
    ])
  );

  return {
    fallbackStage: getStageInstance(normalizeStructure(defaultLayoutPreset.snapshot)),
    getPresentation: (typeId, region) => presentations.get(region)?.get(typeId) ?? null,
    hasDistinctSurface: (typeId, region) =>
      getWidgetById(typeId)?.manifest.graphBearing?.surfaces.includes(region) === true,
  };
};

/**
 * The availability one mounted workbench resolves layouts with: the same object until the session's admin flag
 * changes what the registry offers, so layouts memoized on it stay valid.
 */
export const createLayoutAvailabilitySource = (
  create: () => LayoutAvailability = createRegistryLayoutAvailability
): (() => LayoutAvailability) => {
  let current: { availability: LayoutAvailability; isAdmin: boolean } | undefined;

  return () => {
    const isAdmin = isAdminWidgetSession();

    if (current?.isAdmin !== isAdmin) {
      current = { availability: create(), isAdmin };
    }

    return current.availability;
  };
};
