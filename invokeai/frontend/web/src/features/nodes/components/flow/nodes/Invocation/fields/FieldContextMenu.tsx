import { MenuGroup, MenuItem, MenuList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import {
  IAIContextMenu,
  IAIContextMenuProps,
} from 'common/components/IAIContextMenu';
import { useFieldInputKind } from 'features/nodes/hooks/useFieldInputKind';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import {
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
} from 'features/nodes/store/nodesSlice';
import { MouseEvent, ReactNode, memo, useCallback, useMemo } from 'react';
import { FaMinus, FaPlus } from 'react-icons/fa';
import { menuListMotionProps } from 'theme/components/menu';

type Props = {
  nodeId: string;
  fieldName: string;
  kind: 'input' | 'output';
  children: IAIContextMenuProps<HTMLDivElement>['children'];
};

const FieldContextMenu = ({ nodeId, fieldName, kind, children }: Props) => {
  const dispatch = useAppDispatch();
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, kind);
  const input = useFieldInputKind(nodeId, fieldName);

  const skipEvent = useCallback((e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const isExposed = Boolean(
            nodes.workflow.exposedFields.find(
              (f) => f.nodeId === nodeId && f.fieldName === fieldName
            )
          );

          return { isExposed };
        },
        defaultSelectorOptions
      ),
    [fieldName, nodeId]
  );

  const mayExpose = useMemo(
    () => ['any', 'direct'].includes(input ?? '__UNKNOWN_INPUT__'),
    [input]
  );

  const { isExposed } = useAppSelector(selector);

  const handleExposeField = useCallback(() => {
    dispatch(workflowExposedFieldAdded({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  const handleUnexposeField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  const menuItems = useMemo(() => {
    const menuItems: ReactNode[] = [];
    if (mayExpose && !isExposed) {
      menuItems.push(
        <MenuItem
          key={`${nodeId}.${fieldName}.expose-field`}
          icon={<FaPlus />}
          onClick={handleExposeField}
        >
          Add to Linear View
        </MenuItem>
      );
    }
    if (mayExpose && isExposed) {
      menuItems.push(
        <MenuItem
          key={`${nodeId}.${fieldName}.unexpose-field`}
          icon={<FaMinus />}
          onClick={handleUnexposeField}
        >
          Remove from Linear View
        </MenuItem>
      );
    }
    return menuItems;
  }, [
    fieldName,
    handleExposeField,
    handleUnexposeField,
    isExposed,
    mayExpose,
    nodeId,
  ]);

  return (
    <IAIContextMenu<HTMLDivElement>
      menuProps={{
        size: 'sm',
        isLazy: true,
      }}
      menuButtonProps={{
        bg: 'transparent',
        _hover: { bg: 'transparent' },
      }}
      renderMenu={() =>
        !menuItems.length ? null : (
          <MenuList
            sx={{ visibility: 'visible !important' }}
            motionProps={menuListMotionProps}
            onContextMenu={skipEvent}
          >
            <MenuGroup title={label || fieldTemplateTitle || 'Unknown Field'}>
              {menuItems}
            </MenuGroup>
          </MenuList>
        )
      }
    >
      {children}
    </IAIContextMenu>
  );
};

export default memo(FieldContextMenu);
