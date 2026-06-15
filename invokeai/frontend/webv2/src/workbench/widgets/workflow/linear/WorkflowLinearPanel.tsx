import { Flex, HStack, Icon, SegmentGroup, Splitter } from '@chakra-ui/react';
import { EyeIcon, PencilIcon } from 'lucide-react';
import { useEffect } from 'react';

import { Scrollable } from '@workbench/components/ui/Scrollable';
import { Tabs } from '@workbench/components/ui/Tabs';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { ensureInvocationTemplatesLoaded } from '@workbench/workflows/templates';
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

/** Inspector share of the splitter, as a percentage of the panel height. */
const DEFAULT_INSPECTOR_SIZE_PCT = 35;
const SPLITTER_PANELS = [
  { id: 'content', minSize: 25 },
  { id: 'inspector', minSize: 12 },
];

const getPanelMode = (values: Record<string, unknown>): PanelMode => (values.panelMode === 'edit' ? 'edit' : 'view');

const getEditTab = (values: Record<string, unknown>): EditTab =>
  values.editTab === 'details' || values.editTab === 'json' ? values.editTab : 'form';

const getInspectorSizePct = (values: Record<string, unknown>): number =>
  typeof values.inspectorSizePct === 'number' && Number.isFinite(values.inspectorSizePct)
    ? Math.min(75, Math.max(12, values.inspectorSizePct))
    : DEFAULT_INSPECTOR_SIZE_PCT;

const PanelModeToggle = ({ mode, onChange }: { mode: PanelMode; onChange: (mode: PanelMode) => void }) => {
  type PanelModeItem = {
    label: string;
    icon: typeof EyeIcon;
    mode: PanelMode;
  };

  const PANEL_MODES: PanelModeItem[] = [
    {
      label: 'View',
      icon: EyeIcon,
      mode: 'view',
    },
    {
      label: 'Edit',
      icon: PencilIcon,
      mode: 'edit',
    },
  ];

  const onValueChange = (details: { value: string | null }) => {
    if (details.value === 'view' || details.value === 'edit') {
      onChange(details.value);
    }
  };

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
  const projectGraph = useActiveProjectSelector((project) => project.projectGraph);
  const widgetValues = useActiveProjectSelector((project) => project.widgetStates.workflow.values);
  const dispatch = useWorkbenchDispatch();
  const mode = getPanelMode(widgetValues);
  const editTab = getEditTab(widgetValues);

  useEffect(() => {
    ensureInvocationTemplatesLoaded();
  }, []);

  const patchValues = (values: Record<string, unknown>) =>
    dispatch({ type: 'patchWidgetValues', values, widgetId: 'workflow' });

  return (
    <Flex direction="column" flex="1" h="full" minH="0">
      <HStack flexShrink={0} justify="space-between" px="2" h={10} borderBottomWidth={1}>
        <PanelModeToggle mode={mode} onChange={(panelMode) => patchValues({ panelMode })} />
        {mode === 'edit' ? (
          <Tabs.Root
            size="sm"
            value={editTab}
            variant="outline"
            mb="-1"
            onValueChange={(event) => patchValues({ editTab: event.value })}
          >
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
        <Scrollable flex="1" label="Workflow panel content" minH="0">
          <LinearFormView projectGraph={projectGraph} />
        </Scrollable>
      ) : (
        <Splitter.Root
          defaultSize={[100 - getInspectorSizePct(widgetValues), getInspectorSizePct(widgetValues)]}
          flex="1"
          gap="0"
          minH="0"
          orientation="vertical"
          panels={SPLITTER_PANELS}
          onResizeEnd={(details) => {
            const inspectorSizePct = details.size[1];

            if (typeof inspectorSizePct === 'number') {
              patchValues({ inspectorSizePct });
            }
          }}
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
      )}
    </Flex>
  );
};
