import type { ContextMenuProps } from '@invoke-ai/ui-library';
import { ContextMenu, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFieldInputKind } from 'features/nodes/hooks/useFieldInputKind';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import {
  selectWorkflowSlice,
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
} from 'features/nodes/store/workflowSlice';
import type { ReactNode } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMinusBold, PiPlusBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
  kind: 'input' | 'output';
  children: ContextMenuProps<HTMLDivElement>['children'];
};

const FieldContextMenu = ({ nodeId, fieldName, kind, children }: Props) => {
  const dispatch = useAppDispatch();
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, kind);
  const input = useFieldInputKind(nodeId, fieldName);
  const { t } = useTranslation();

  const selectIsExposed = useMemo(
    () =>
      createSelector(selectWorkflowSlice, (workflow) => {
        return Boolean(workflow.exposedFields.find((f) => f.nodeId === nodeId && f.fieldName === fieldName));
      }),
    [fieldName, nodeId]
  );

  const mayExpose = useMemo(() => input && ['any', 'direct'].includes(input), [input]);

  const isExposed = useAppSelector(selectIsExposed);

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
        <MenuItem key={`${nodeId}.${fieldName}.expose-field`} icon={<PiPlusBold />} onClick={handleExposeField}>
          {t('nodes.addLinearView')}
        </MenuItem>
      );
    }
    if (mayExpose && isExposed) {
      menuItems.push(
        <MenuItem key={`${nodeId}.${fieldName}.unexpose-field`} icon={<PiMinusBold />} onClick={handleUnexposeField}>
          {t('nodes.removeLinearView')}
        </MenuItem>
      );
    }
    return menuItems;
  }, [fieldName, handleExposeField, handleUnexposeField, isExposed, mayExpose, nodeId, t]);

  const renderMenuFunc = useCallback(
    () =>
      !menuItems.length ? null : (
        <MenuList visibility="visible">
          <MenuGroup title={label || fieldTemplateTitle || t('nodes.unknownField')}>{menuItems}</MenuGroup>
        </MenuList>
      ),
    [fieldTemplateTitle, label, menuItems, t]
  );

  return <ContextMenu renderMenu={renderMenuFunc}>{children}</ContextMenu>;
};

export default memo(FieldContextMenu);
