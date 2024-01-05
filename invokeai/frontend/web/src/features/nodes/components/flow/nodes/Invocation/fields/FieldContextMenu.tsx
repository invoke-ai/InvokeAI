import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { InvContextMenuProps } from 'common/components/InvContextMenu/InvContextMenu';
import { InvContextMenu } from 'common/components/InvContextMenu/InvContextMenu';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvMenuList } from 'common/components/InvMenu/InvMenuList';
import { InvMenuGroup } from 'common/components/InvMenu/wrapper';
import { useFieldInputKind } from 'features/nodes/hooks/useFieldInputKind';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import {
  selectWorkflowSlice,
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
} from 'features/nodes/store/workflowSlice';
import type { MouseEvent, ReactNode } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaMinus, FaPlus } from 'react-icons/fa';

type Props = {
  nodeId: string;
  fieldName: string;
  kind: 'input' | 'output';
  children: InvContextMenuProps<HTMLDivElement>['children'];
};

const FieldContextMenu = ({ nodeId, fieldName, kind, children }: Props) => {
  const dispatch = useAppDispatch();
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, kind);
  const input = useFieldInputKind(nodeId, fieldName);
  const { t } = useTranslation();

  const skipEvent = useCallback((e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const selector = useMemo(
    () =>
      createMemoizedSelector(selectWorkflowSlice, (workflow) => {
        const isExposed = Boolean(
          workflow.exposedFields.find(
            (f) => f.nodeId === nodeId && f.fieldName === fieldName
          )
        );

        return { isExposed };
      }),
    [fieldName, nodeId]
  );

  const mayExpose = useMemo(
    () => input && ['any', 'direct'].includes(input),
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
        <InvMenuItem
          key={`${nodeId}.${fieldName}.expose-field`}
          icon={<FaPlus />}
          onClick={handleExposeField}
        >
          {t('nodes.addLinearView')}
        </InvMenuItem>
      );
    }
    if (mayExpose && isExposed) {
      menuItems.push(
        <InvMenuItem
          key={`${nodeId}.${fieldName}.unexpose-field`}
          icon={<FaMinus />}
          onClick={handleUnexposeField}
        >
          {t('nodes.removeLinearView')}
        </InvMenuItem>
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
    t,
  ]);

  const renderMenuFunc = useCallback(
    () =>
      !menuItems.length ? null : (
        <InvMenuList visibility="visible" onContextMenu={skipEvent}>
          <InvMenuGroup
            title={label || fieldTemplateTitle || t('nodes.unknownField')}
          >
            {menuItems}
          </InvMenuGroup>
        </InvMenuList>
      ),
    [fieldTemplateTitle, label, menuItems, skipEvent, t]
  );

  return (
    <InvContextMenu renderMenu={renderMenuFunc}>{children}</InvContextMenu>
  );
};

export default memo(FieldContextMenu);
