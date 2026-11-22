import type { InvocationRoute, InvocationSourceId } from '@workbench/invocationContracts';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import type { IsCenterView, SavedWorkspace, WorkspacePlacement, WorkspaceSources } from './workspaceDerivations';

import { getEntryStage, getWorkspaceHomeFronts, getWorkspaceJob } from './workspaceDerivations';

/** What entering a workspace changes: region fronts and the route. Nothing else is written. */
export interface WorkspaceEntry {
  fronts: Partial<Record<WidgetRegion, WidgetInstanceId>>;
  invocation: InvocationRoute;
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
  const home = SIDE_REGION_ORDER.find(
    (region) => runsJob(homeFronts[region]) && placement.widgetRegions[region].instanceIds.includes(homeFronts[region])
  );

  if (home) {
    return { instanceId: homeFronts[home], region: home };
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
 * Entering a workspace puts its job on screen: the center fronts the stage (Away ends), the job's source is fronted in
 * its region, and the route becomes the job, each lock kept. A collapsed or closed panel stays so, and sizes, other
 * fronts and windows are untouched. A workspace with no job leaves the route as it was.
 */
export const enterWorkspace = ({
  invocation,
  isCenterView,
  placement,
  sources,
  workspace,
}: {
  invocation: InvocationRoute;
  isCenterView: IsCenterView;
  /** What is laid out for this project now. */
  placement: WorkspacePlacement;
  sources: WorkspaceSources;
  workspace: SavedWorkspace;
}): WorkspaceEntry => {
  const fronts: WorkspaceEntry['fronts'] = {};
  const stage = getEntryStage(workspace, placement, isCenterView);

  if (stage !== null && placement.widgetRegions.center.activeInstanceId !== stage) {
    fronts.center = stage;
  }

  const job = getWorkspaceJob(workspace, placement, sources);

  if (!job) {
    return { fronts, invocation };
  }

  // Side regions only: a source in the center is the stage itself, or the stage outranks it.
  const source = findJobSourceFront(workspace, placement, job.sourceId, sources.getSourceId);

  if (source && placement.widgetRegions[source.region].activeInstanceId !== source.instanceId) {
    fronts[source.region] = source.instanceId;
  }

  const sourceId = invocation.sourceLocked ? invocation.sourceId : job.sourceId;
  const destination = invocation.destinationLocked ? invocation.destination : job.destination;

  return {
    fronts,
    invocation:
      sourceId === invocation.sourceId && destination === invocation.destination
        ? invocation
        : { ...invocation, destination, sourceId },
  };
};
