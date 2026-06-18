import type {
  GraphBearingSurfaceContract,
  GraphContract,
  Project,
  WidgetInstanceContract,
  WidgetManifest,
  WidgetRuntimeApi,
  WorkbenchRegion,
} from '@workbench/types';

import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { IconButton } from '@workbench/components/ui';
import { GraphPreviewDialog } from '@workbench/graph-preview';
import { createGraphBearingSurface } from '@workbench/graphSurfaces';
import { useActiveProject, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { compileProjectGraph } from '@workbench/workflows/buildGraph';
import { getInvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import { GitBranchIcon, MoreHorizontalIcon, TargetIcon } from 'lucide-react';
import { useState } from 'react';

/**
 * The widget frame's shared header actions menu. It hosts the universal
 * graph-bearing actions (`Set Source`, `View Graph`) and any extra entries the
 * widget's manifest contributes via `headerMenu` — one menu per widget, so
 * widgets extend the frame instead of stacking their own menus and toolbars.
 */

const getPreviewGraph = (project: Project, surface: GraphBearingSurfaceContract): GraphContract | null => {
  // The project graph compiles fresh for preview so `View Graph` always shows
  // what Invoke would run right now; widget graphs keep their last compile.
  if (surface.sourceId === 'project-graph') {
    const templatesSnapshot = getInvocationTemplatesSnapshot();

    if (templatesSnapshot.status !== 'loaded') {
      return null;
    }

    try {
      return compileProjectGraph(project.projectGraph, templatesSnapshot.templates);
    } catch {
      return null;
    }
  }

  return project.widgetGraphs[surface.widgetId] ?? null;
};

const GraphSurfaceMenuItems = ({
  surface,
  onPreview,
}: {
  surface: GraphBearingSurfaceContract;
  onPreview: () => void;
}) => {
  const activeProject = useActiveProject();
  const dispatch = useWorkbenchDispatch();
  const isActiveSource = activeProject.invocation.sourceId === surface.sourceId;

  return (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
        Graph
      </Menu.ItemGroupLabel>
      <Menu.Item
        value="set-source"
        disabled={isActiveSource || !surface.canSetSource}
        _disabled={{ opacity: 0.4 }}
        onClick={() => dispatch({ sourceId: surface.sourceId, type: 'setInvocationSource' })}
      >
        <Icon as={TargetIcon} boxSize="3.5" />
        <Menu.ItemText>Set Source</Menu.ItemText>
        {isActiveSource ? (
          <Text color="fg.subtle" fontSize="2xs" ms="auto">
            Active
          </Text>
        ) : null}
      </Menu.Item>
      <Menu.Item value="view-graph" disabled={!surface.canPreviewGraph} onClick={onPreview}>
        <Icon as={GitBranchIcon} boxSize="3.5" />
        <Menu.ItemText>View Graph</Menu.ItemText>
      </Menu.Item>
    </Menu.ItemGroup>
  );
};

export const WidgetActionsMenu = ({
  instance,
  manifest,
  region,
  runtime,
}: {
  instance: WidgetInstanceContract;
  manifest: WidgetManifest;
  region: WorkbenchRegion;
  runtime: WidgetRuntimeApi;
}) => {
  const activeProject = useActiveProject();
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const surface = manifest.graphBearing?.surfaces.includes(region) ? createGraphBearingSurface(manifest, region) : null;
  const HeaderMenu = manifest.headerMenu;

  if (!surface && !HeaderMenu) {
    return null;
  }

  const previewGraph = surface && isPreviewOpen ? getPreviewGraph(activeProject, surface) : null;
  // The project graph mirrors the editable document, so the preview can reuse
  // the editor's node positions instead of auto-layouting.
  const positionHints =
    surface?.sourceId === 'project-graph'
      ? Object.fromEntries(activeProject.projectGraph.nodes.map((node) => [node.id, node.position]))
      : undefined;

  return (
    <>
      <Menu.Root positioning={{ placement: 'bottom-end' }}>
        <Menu.Trigger asChild>
          <IconButton aria-label={`${manifest.labelText} actions`} color="fg.muted" size="2xs" variant="ghost">
            <MoreHorizontalIcon />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <Menu.Content minW="13rem">
              {surface ? <GraphSurfaceMenuItems surface={surface} onPreview={() => setIsPreviewOpen(true)} /> : null}
              {surface && HeaderMenu ? <Menu.Separator borderColor="border.subtle" /> : null}
              {HeaderMenu ? (
                <HeaderMenu instance={instance} manifest={manifest} region={region} runtime={runtime} />
              ) : null}
            </Menu.Content>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      {surface ? (
        <GraphPreviewDialog
          graph={previewGraph}
          graphId={surface.graphId}
          isOpen={isPreviewOpen}
          positionHints={positionHints}
          sourceId={surface.sourceId}
          title={surface.label}
          onOpenChange={setIsPreviewOpen}
        />
      ) : null}
    </>
  );
};
