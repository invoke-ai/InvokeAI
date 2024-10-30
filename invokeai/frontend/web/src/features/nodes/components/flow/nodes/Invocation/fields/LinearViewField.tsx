import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { attachClosestEdge, extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { Box, Circle, Flex, Icon, IconButton, Spacer, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { DndState } from 'features/dnd/dnd';
import { Dnd, idle } from 'features/dnd/dnd';
import { DndDropIndicator } from 'features/dnd/DndDropIndicator';
import { InvocationInputFieldCheck } from 'features/nodes/components/flow/nodes/Invocation/fields/InvocationFieldCheck';
import { useFieldOriginalValue } from 'features/nodes/hooks/useFieldOriginalValue';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { workflowExposedFieldRemoved } from 'features/nodes/store/workflowSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiInfoBold, PiTrashSimpleBold } from 'react-icons/pi';

import EditableFieldTitle from './EditableFieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

type Props = {
  nodeId: string;
  fieldName: string;
};

const LinearViewFieldInternal = ({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { isValueChanged, onReset } = useFieldOriginalValue(nodeId, fieldName);
  const { isMouseOverNode, handleMouseOut, handleMouseOver } = useMouseOverNode(nodeId);
  const { t } = useTranslation();

  const handleRemoveField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  const ref = useRef<HTMLDivElement>(null);
  const [dndState, setDndState] = useState<DndState>(idle);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      draggable({
        element,
        getInitialData() {
          return Dnd.Source.singleWorkflowField.getData({ fieldIdentifier: { nodeId, fieldName } });
        },
        onDragStart() {
          setDndState({ type: 'is-dragging' });
        },
        onDrop() {
          setDndState(idle);
        },
      }),
      dropTargetForElements({
        element,
        canDrop({ source }) {
          if (!Dnd.Source.singleWorkflowField.typeGuard(source.data)) {
            return false;
          }
          return true;
        },
        getData({ input }) {
          const data = Dnd.Source.singleWorkflowField.getData({ fieldIdentifier: { nodeId, fieldName } });
          return attachClosestEdge(data, {
            element,
            input,
            allowedEdges: ['top', 'bottom'],
          });
        },
        getIsSticky() {
          return true;
        },
        onDragEnter({ self }) {
          const closestEdge = extractClosestEdge(self.data);
          setDndState({ type: 'is-dragging-over', closestEdge });
        },
        onDrag({ self }) {
          const closestEdge = extractClosestEdge(self.data);

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setDndState((current) => {
            if (current.type === 'is-dragging-over' && current.closestEdge === closestEdge) {
              return current;
            }
            return { type: 'is-dragging-over', closestEdge };
          });
        },
        onDragLeave() {
          setDndState(idle);
        },
        onDrop() {
          setDndState(idle);
        },
      })
    );
  }, [fieldName, nodeId]);

  return (
    <Box position="relative" w="full">
      <Flex
        ref={ref}
        // This is used to trigger the post-move flash animation
        data-field-name={fieldName}
        onMouseEnter={handleMouseOver}
        onMouseLeave={handleMouseOut}
        layerStyle="second"
        alignItems="center"
        position="relative"
        borderRadius="base"
        w="full"
        p={2}
      >
        <Flex flexDir="column" w="full">
          <Flex alignItems="center" gap={2}>
            <EditableFieldTitle nodeId={nodeId} fieldName={fieldName} kind="inputs" />
            <Spacer />
            {isMouseOverNode && <Circle size={2} borderRadius="full" bg="invokeBlue.500" />}
            {isValueChanged && (
              <IconButton
                aria-label={t('nodes.resetToDefaultValue')}
                tooltip={t('nodes.resetToDefaultValue')}
                variant="ghost"
                size="sm"
                onClick={onReset}
                icon={<PiArrowCounterClockwiseBold />}
              />
            )}
            <Tooltip
              label={<FieldTooltipContent nodeId={nodeId} fieldName={fieldName} kind="inputs" />}
              openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
              placement="top"
            >
              <Flex h="full" alignItems="center">
                <Icon fontSize="sm" color="base.300" as={PiInfoBold} />
              </Flex>
            </Tooltip>
            <IconButton
              aria-label={t('nodes.removeLinearView')}
              tooltip={t('nodes.removeLinearView')}
              variant="ghost"
              size="sm"
              onClick={handleRemoveField}
              icon={<PiTrashSimpleBold />}
            />
          </Flex>
          <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
        </Flex>
      </Flex>
      {dndState.type === 'is-dragging-over' && dndState.closestEdge ? (
        <DndDropIndicator
          edge={dndState.closestEdge}
          // This is the gap between items in the list
          gap="var(--invoke-space-2)"
        />
      ) : null}
    </Box>
  );
};

const LinearViewField = ({ nodeId, fieldName }: Props) => {
  return (
    <InvocationInputFieldCheck nodeId={nodeId} fieldName={fieldName}>
      <LinearViewFieldInternal nodeId={nodeId} fieldName={fieldName} />
    </InvocationInputFieldCheck>
  );
};

export default memo(LinearViewField);
