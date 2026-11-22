import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import type { LayoutAvailability, ResolvedLayout } from './resolveLayout';

import { SIDE_REGIONS } from './structure';

/** A type a shown widget acts on that is not on screen, and whether this workspace places it (else it is elsewhere). */
export interface UnmetNeed {
  typeId: WidgetTypeId;
  isPlacedHere: boolean;
}

type NeedsLayout = Pick<ResolvedLayout, 'floatingWidgets' | 'guests' | 'widgetInstances' | 'widgetRegions'>;

/** What is on screen: the center's front, each open side panel's front, and every window not shaded. */
const getOnScreenInstanceIds = (layout: NeedsLayout): Set<WidgetInstanceId> => {
  const onScreen = new Set<WidgetInstanceId>([layout.widgetRegions.center.activeInstanceId]);

  for (const region of SIDE_REGIONS) {
    const { activeInstanceId, instanceIds, isCollapsed } = layout.widgetRegions[region];

    if (!isCollapsed && instanceIds.includes(activeInstanceId)) {
      onScreen.add(activeInstanceId);
    }
  }
  for (const [instanceId, window] of Object.entries(layout.floatingWidgets ?? {})) {
    if (window.mode !== 'shaded') {
      onScreen.add(instanceId);
    }
  }

  return onScreen;
};

const placesHere = (layout: NeedsLayout, typeId: WidgetTypeId): boolean => {
  const isType = (instanceId: WidgetInstanceId) =>
    layout.widgetInstances[instanceId]?.typeId === typeId && layout.guests[instanceId] === undefined;

  return (
    Object.values(layout.widgetRegions).some(({ instanceIds }) => instanceIds.some(isType)) ||
    Object.keys(layout.floatingWidgets ?? {}).some(isType)
  );
};

/**
 * The first of an on-screen widget's `needs` that is not on screen, or null: none for a widget that is itself hidden,
 * and none for a need this session cannot show at all.
 */
export const getUnmetNeed = (
  layout: NeedsLayout,
  instanceId: WidgetInstanceId,
  needs: readonly WidgetTypeId[],
  getPresentation: LayoutAvailability['getPresentation']
): UnmetNeed | null => {
  const onScreen = getOnScreenInstanceIds(layout);

  if (!onScreen.has(instanceId)) {
    return null;
  }

  const typesOnScreen = new Set([...onScreen].map((id) => layout.widgetInstances[id]?.typeId));
  const typeId = needs.find(
    (need) =>
      !typesOnScreen.has(need) &&
      (['center', 'left', 'right', 'bottom'] as const).some((region) => getPresentation(need, region) === 'view')
  );

  return typeId === undefined ? null : { isPlacedHere: placesHere(layout, typeId), typeId };
};
