import { Flex, Stack } from '@chakra-ui/react';
import { Scrollable } from '@platform/ui/Scrollable';
import { useCanvasEngine, type CanvasEngineHandle } from '@workbench/canvas-operations/react';
import { isCanvasInteractionLocked } from '@workbench/widgets/canvas/canvasInteractionLock';
import { useCanvasActiveTool, useCanvasOperation } from '@workbench/widgets/canvas/engineStoreHooks';
import { PropertyGroup } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import {
  OPERATION_PRESENTATION_ADAPTERS,
  TOOL_PRESENTATION_ADAPTERS,
} from '@workbench/widgets/canvas/tool-presentation/toolAdapters';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useLayoutEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { GroupSelectedNotice } from './GroupSelectedNotice';
import { LayerSection } from './LayerSection';
import { PropertiesSection } from './PropertiesSection';

const OPERATION_TITLE_KEYS = {
  filter: 'widgets.layers.rasterFilter.title',
  'select-object': 'widgets.layers.selectObject.title',
} as const;

/** Show active operation before tool forms; use existing engine stores/transactions without mirrored state. */
export const PropertiesPane = () => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();
  const isSurfaceInteractionLocked = useActiveProjectSelector((project) =>
    isCanvasInteractionLocked(project.canvas, project.queue.items)
  );

  if (!engine) {
    return (
      <Flex align="center" color="fg.muted" fontSize="md" h="full" justify="center" p="4">
        {t('widgets.properties.noCanvas')}
      </Flex>
    );
  }

  return (
    <Scrollable h="full">
      <ConnectedProperties engine={engine} isSurfaceInteractionLocked={isSurfaceInteractionLocked} />
    </Scrollable>
  );
};

const ConnectedProperties = ({
  engine,
  isSurfaceInteractionLocked,
}: {
  engine: CanvasEngineHandle;
  isSurfaceInteractionLocked: boolean;
}) => {
  const { t } = useTranslation();
  const activeTool = useCanvasActiveTool(engine);
  const operation = useCanvasOperation(engine);
  const running = operation.status === 'active' ? OPERATION_PRESENTATION_ADAPTERS[operation.identity.kind] : null;
  const tool = TOOL_PRESENTATION_ADAPTERS[activeTool];
  const toolName = t(`widgets.canvas.tools.${tool.id}`);
  const Preview = tool.preview;
  const regionProps = { engine, isSurfaceInteractionLocked };
  // The pane has ONE sticky footer: a running operation's, else the tool
  // form's; the operation always wins because the tool section is inert then.
  const Footer = running ? running.footer : tool.footer;

  // `inert` will blur a focused tool control once an operation starts; hand focus to the operation's Cancel first.
  const root = useRef<HTMLDivElement>(null);
  const toolSection = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (running && toolSection.current?.contains(document.activeElement)) {
      root.current?.querySelector<HTMLElement>('[data-pane-action="cancel"]')?.focus();
    }
  }, [running]);

  return (
    <Stack ref={root} gap="0">
      {running ? (
        <PropertiesSection
          subtitle={t(OPERATION_TITLE_KEYS[running.kind])}
          title={t('widgets.properties.sections.operation')}
        >
          {running.groups.map((group) => (
            <PropertyGroup key={group.id} collapsible={group.collapsible} id={group.id} label={t(group.labelKey)}>
              <group.body {...regionProps} />
            </PropertyGroup>
          ))}
        </PropertiesSection>
      ) : null}
      <PropertiesSection
        ref={toolSection}
        disabled={isSurfaceInteractionLocked || running !== null}
        subtitle={toolName}
        title={t('widgets.properties.sections.tool')}
      >
        {tool.paintsLeaf && !running ? <GroupSelectedNotice /> : null}
        {Preview ? <Preview engine={engine} isExternalInteractionLocked={isSurfaceInteractionLocked} /> : null}
        {tool.groups.map((group) => (
          // Keyed by GROUP id, not tool id: tools sharing a group keep its
          // DOM (and collapse state) alive across the tool switch.
          <PropertyGroup key={group.id} collapsible={group.collapsible} id={group.id} label={t(group.labelKey)}>
            <group.body {...regionProps} />
          </PropertyGroup>
        ))}
      </PropertiesSection>
      <LayerSection disabled={isSurfaceInteractionLocked || running !== null} />
      {Footer ? (
        <Flex
          bg="bg.panel"
          borderColor="border.subtle"
          borderTopWidth="1px"
          bottom="0"
          position="sticky"
          px="3"
          py="1.5"
        >
          <Footer engine={engine} isExternalInteractionLocked={isSurfaceInteractionLocked} />
        </Flex>
      ) : null}
    </Stack>
  );
};
