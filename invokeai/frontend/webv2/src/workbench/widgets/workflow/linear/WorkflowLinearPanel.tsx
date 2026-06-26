import { Flex, HStack, Icon, SegmentGroup, Splitter } from '@chakra-ui/react';
import { Scrollable, Tabs } from '@workbench/components/ui';
import { useActiveProjectSelector, useWidgetValuesSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { ensureInvocationTemplatesLoaded } from '@workbench/workflows/templates';
import { EyeIcon, PencilIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo } from 'react';

import { FormBuilderTab } from './FormBuilderTab';
import { LinearFormView } from './LinearFormView';
import { NodeInspector } from './NodeInspector';
import { WorkflowDetailsTab } from './WorkflowDetailsTab';
import { WorkflowJsonTab } from './WorkflowJsonTab';

/**
 * The workflow widget's side panel — the Linear UI. View mode runs the
 * project graph through its form; Edit mode opens the legacy-style tab set
 * (Form builder / Details / JSON) with the node inspector pinned below.
 * Panel mode, tab, and inspector height live in the widget's own state.
 */

type PanelMode = 'view' | 'edit';
type EditTab = 'form' | 'details' | 'json';
type PanelModeItem = {
  label: string;
  icon: typeof EyeIcon;
  mode: PanelMode;
};

/** Inspector share of the splitter, as a percentage of the panel height. */
const DEFAULT_INSPECTOR_SIZE_PCT = 35;
const SPLITTER_PANELS = [
  { id: 'content', minSize: 25 },
  { id: 'inspector', minSize: 12 },
];
const PANEL_MODES: PanelModeItem[] = [
  { label: 'View', icon: EyeIcon, mode: 'view' },
  { label: 'Edit', icon: PencilIcon, mode: 'edit' },
];

export interface WorkflowPanelState {
  editTab: EditTab;
  inspectorSizePct: number;
  mode: PanelMode;
}

const getPanelMode = (values: Record<string, unknown>): PanelMode => (values.panelMode === 'edit' ? 'edit' : 'view');

const getEditTab = (values: Record<string, unknown>): EditTab =>
  values.editTab === 'details' || values.editTab === 'json' ? values.editTab : 'form';

const getInspectorSizePct = (values: Record<string, unknown>): number =>
  typeof values.inspectorSizePct === 'number' && Number.isFinite(values.inspectorSizePct)
    ? Math.min(75, Math.max(12, values.inspectorSizePct))
    : DEFAULT_INSPECTOR_SIZE_PCT;

export const getWorkflowPanelState = (values: Record<string, unknown>): WorkflowPanelState => ({
  editTab: getEditTab(values),
  inspectorSizePct: getInspectorSizePct(values),
  mode: getPanelMode(values),
});

export const areWorkflowPanelStatesEqual = (left: WorkflowPanelState, right: WorkflowPanelState): boolean =>
  left.mode === right.mode && left.editTab === right.editTab && left.inspectorSizePct === right.inspectorSizePct;

const PanelModeToggle = ({ mode, onChange }: { mode: PanelMode; onChange: (mode: PanelMode) => void }) => {
  const onValueChange = useCallback(
    (details: { value: string | null }) => {
      if (details.value === 'view' || details.value === 'edit') {
        onChange(details.value);
      }
    },
    [onChange]
  );

  return (
    <SegmentGroup.Root value={mode} onValueChange={onValueChange} size="xs">
      <SegmentGroup.Indicator />
      {PANEL_MODES.map(({ label, icon, mode }) => (
        <SegmentGroup.Item key={mode} value={mode}>
          <SegmentGroup.ItemHiddenInput />
          <Icon as={icon} boxSize="3" />
          <SegmentGroup.ItemText>{label}</SegmentGroup.ItemText>
        </SegmentGroup.Item>
      ))}
    </SegmentGroup.Root>
  );
};

export const WorkflowLinearPanel = () => {
  const { editTab, inspectorSizePct, mode } = useWidgetValuesSelector(
    'workflow',
    getWorkflowPanelState,
    areWorkflowPanelStatesEqual
  );
  const dispatch = useWorkbenchDispatch();

  useEffect(() => {
    ensureInvocationTemplatesLoaded();
  }, []);

  const patchValues = useCallback(
    (values: Record<string, unknown>) => dispatch({ type: 'patchWidgetValues', values, widgetId: 'workflow' }),
    [dispatch]
  );
  const onPanelModeChange = useCallback((panelMode: PanelMode) => patchValues({ panelMode }), [patchValues]);
  const onEditTabChange = useCallback(
    (event: { value: string }) => patchValues({ editTab: event.value }),
    [patchValues]
  );

  return (
    <Flex direction="column" flex="1" h="full" minH="0">
      <HStack flexShrink={0} justify="space-between" px="2" h={10} borderBottomWidth={1}>
        <PanelModeToggle mode={mode} onChange={onPanelModeChange} />
        {mode === 'edit' ? (
          <Tabs.Root size="sm" value={editTab} variant="outline" mb="-1" onValueChange={onEditTabChange}>
            <Tabs.List>
              <Tabs.Trigger value="form" fontSize="2xs">
                Form
              </Tabs.Trigger>
              <Tabs.Trigger value="details" fontSize="2xs">
                Details
              </Tabs.Trigger>
              <Tabs.Trigger value="json" fontSize="2xs">
                JSON
              </Tabs.Trigger>
            </Tabs.List>
          </Tabs.Root>
        ) : null}
      </HStack>
      {mode === 'view' ? (
        <WorkflowLinearViewContent />
      ) : (
        <WorkflowLinearEditContent editTab={editTab} inspectorSizePct={inspectorSizePct} patchValues={patchValues} />
      )}
    </Flex>
  );
};

const WorkflowLinearViewContent = () => {
  const projectGraph = useActiveProjectSelector((project) => project.projectGraph);

  return (
    <Scrollable flex="1" label="Workflow panel content" minH="0">
      <LinearFormView projectGraph={projectGraph} />
    </Scrollable>
  );
};

const WorkflowLinearEditContent = ({
  editTab,
  inspectorSizePct,
  patchValues,
}: {
  editTab: EditTab;
  inspectorSizePct: number;
  patchValues: (values: Record<string, unknown>) => void;
}) => {
  const projectGraph = useActiveProjectSelector((project) => project.projectGraph);
  const defaultSize = useMemo(() => [100 - inspectorSizePct, inspectorSizePct], [inspectorSizePct]);
  const onResizeEnd = useCallback(
    (details: { size: number[] }) => {
      const inspectorSizePct = details.size[1];

      if (typeof inspectorSizePct === 'number') {
        patchValues({ inspectorSizePct });
      }
    },
    [patchValues]
  );

  return (
    <Splitter.Root
      defaultSize={defaultSize}
      flex="1"
      gap="0"
      minH="0"
      orientation="vertical"
      panels={SPLITTER_PANELS}
      onResizeEnd={onResizeEnd}
    >
      <Splitter.Panel id="content" minH="0" minW="0" overflow="hidden">
        {editTab === 'json' ? (
          <WorkflowJsonTab projectGraph={projectGraph} />
        ) : (
          <Scrollable flex="1" h="full" label="Workflow panel content" minH="0" minW="0" w="full">
            {editTab === 'form' ? (
              <FormBuilderTab projectGraph={projectGraph} />
            ) : (
              <WorkflowDetailsTab metadata={projectGraph} />
            )}
          </Scrollable>
        )}
      </Splitter.Panel>
      <Splitter.ResizeTrigger aria-label="Resize node inspector" id="content:inspector" />
      <Splitter.Panel id="inspector" minH="0" overflow="hidden">
        <NodeInspector projectGraph={projectGraph} />
      </Splitter.Panel>
    </Splitter.Root>
  );
};
