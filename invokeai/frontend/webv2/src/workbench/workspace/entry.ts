import type { InvocationRoute, InvocationSourceId } from '@workbench/invocationContracts';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import type { WorkspaceStructure } from './structure';
import type { SurfaceAvailability, WorkspaceView } from './view';
import type { SavedWorkspace, WorkspacePlacement, WorkspaceSources } from './workspaceDerivations';

import { frontRegion } from './view';
import { getRegionFront, getWorkspaceHomeFronts, getWorkspaceJob } from './workspaceDerivations';

/** What entering a workspace changes: the project's view of it and the route. Nothing else is written. */
export interface WorkspaceEntry {
  invocation: InvocationRoute;
  view: WorkspaceView | undefined;
}

const SIDE_REGION_ORDER: readonly WidgetRegion[] = ['left', 'right', 'bottom'];

/** The side-region instance that runs the job: the saved home front while still placed there, else the first placed. */
const findJobSourceFront = (
  workspace: SavedWorkspace,
  placement: WorkspacePlacement,
  sourceId: InvocationSourceId,
  getSourceId: WorkspaceSources['getSourceId']
): { instanceId: WidgetInstanceId; region: WidgetRegion } | null => {
  const runsJob = (instanceId: WidgetInstanceId) => {
    const typeId = placement.widgetInstances[instanceId]?.typeId;

    return typeId !== undefined && getSourceId(typeId) === sourceId;
  };
  const homeFronts = getWorkspaceHomeFronts(workspace.snapshot);

  for (const region of SIDE_REGION_ORDER) {
    const home = homeFronts[region];

    if (home !== null && runsJob(home) && placement.widgetRegions[region].instanceIds.includes(home)) {
      return { instanceId: home, region };
    }
  }

  for (const region of SIDE_REGION_ORDER) {
    const { activeInstanceId, instanceIds } = placement.widgetRegions[region];
    const members = instanceIds.filter(runsJob);
    const instanceId = members.includes(activeInstanceId) ? activeInstanceId : members[0];

    if (instanceId !== undefined) {
      return { instanceId, region };
    }
  }

  return null;
};

/**
 * Entering a workspace puts its job on screen, writing view only: the stage is fronted in the center (`frontRegion`,
 * so Away ends and a side region's explicit front of the stage's own instance goes: the stage is in the center after
 * every entry), the job's source is fronted in its region, and the route becomes the job, each lock kept. A collapsed
 * panel stays so, and sizes, other fronts and windows are untouched. A workspace with no job leaves the route as it was.
 */
export const enterWorkspace = ({
  invocation,
  resolve,
  sources,
  structure,
  surfaces,
  view,
  workspace,
}: {
  invocation: InvocationRoute;
  /** What the structure shows seen through a view. */
  resolve: (view: WorkspaceView | undefined) => WorkspacePlacement;
  sources: WorkspaceSources;
  structure: WorkspaceStructure;
  surfaces: SurfaceAvailability;
  view: WorkspaceView | undefined;
  workspace: SavedWorkspace;
}): WorkspaceEntry => {
  const stage = getRegionFront(structure.widgetRegions.center);
  const home = stage === null ? view : frontRegion(view, structure, 'center', stage, surfaces);
  const placement = resolve(home);
  const job = getWorkspaceJob(workspace, placement, sources);

  if (!job) {
    return { invocation, view: home };
  }

  // Side regions only: a source in the center is the stage itself, or the stage outranks it.
  const source = findJobSourceFront(workspace, placement, job.sourceId, sources.getSourceId);
  const sourceId = invocation.sourceLocked ? invocation.sourceId : job.sourceId;
  const destination = invocation.destinationLocked ? invocation.destination : job.destination;

  return {
    invocation:
      sourceId === invocation.sourceId && destination === invocation.destination
        ? invocation
        : { ...invocation, destination, sourceId },
    // The stage's own instance in a side region stays behind the center, where it already shows.
    view:
      source && source.instanceId !== stage
        ? frontRegion(home, structure, source.region, source.instanceId, surfaces)
        : home,
  };
};
