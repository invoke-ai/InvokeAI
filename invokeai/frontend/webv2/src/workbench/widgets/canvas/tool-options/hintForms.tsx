import type { ToolFormProps, ToolPropertyForm } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { HintCard } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ViewHints = (_props: ToolFormProps) => {
  const { t } = useTranslation();
  const rows = useMemo(
    () => [
      { effect: t('widgets.properties.hints.view.drag'), gesture: t('widgets.properties.gestures.drag') },
      { effect: t('widgets.properties.hints.view.scroll'), gesture: t('widgets.properties.gestures.scroll') },
      { effect: t('widgets.properties.hints.view.hold'), gesture: t('widgets.properties.gestures.holdSpace') },
    ],
    [t]
  );
  return <HintCard rows={rows} />;
};

const ColorPickerHints = (_props: ToolFormProps) => {
  const { t } = useTranslation();
  const rows = useMemo(
    () => [
      { effect: t('widgets.properties.hints.colorPicker.click'), gesture: t('widgets.properties.gestures.click') },
      { effect: t('widgets.properties.hints.colorPicker.hold'), gesture: t('widgets.properties.gestures.holdAlt') },
    ],
    [t]
  );
  return <HintCard rows={rows} />;
};

const SamToolHints = (_props: ToolFormProps) => {
  const { t } = useTranslation();
  const rows = useMemo(
    () => [
      { effect: t('widgets.properties.hints.sam.click'), gesture: t('widgets.properties.gestures.click') },
      { effect: t('widgets.properties.hints.sam.shiftClick'), gesture: t('widgets.properties.gestures.shiftClick') },
      { effect: t('widgets.properties.hints.sam.clickPoint'), gesture: t('widgets.properties.gestures.clickPoint') },
      { effect: t('widgets.properties.hints.sam.drag'), gesture: t('widgets.properties.gestures.drag') },
    ],
    [t]
  );
  return <HintCard rows={rows} />;
};

export const viewForm: ToolPropertyForm = {
  groups: [{ body: ViewHints, id: 'view-gestures', labelKey: 'widgets.properties.groups.gestures' }],
  id: 'view',
};

export const colorPickerForm: ToolPropertyForm = {
  groups: [{ body: ColorPickerHints, id: 'color-picker-gestures', labelKey: 'widgets.properties.groups.gestures' }],
  id: 'colorPicker',
};

export const samToolForm: ToolPropertyForm = {
  groups: [{ body: SamToolHints, id: 'sam-gestures', labelKey: 'widgets.properties.groups.gestures' }],
  id: 'sam',
};
