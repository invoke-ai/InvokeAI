export type RoutingRelationship = 'arrow' | 'link';

export const getRoutingRelationship = (locks: {
  destinationLocked: boolean;
  sourceLocked: boolean;
}): RoutingRelationship => (locks.destinationLocked || locks.sourceLocked ? 'link' : 'arrow');
